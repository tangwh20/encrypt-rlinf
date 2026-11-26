#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <ftw.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zlib.h>
#ifdef RLINF_SEALED_MODULES
#include <openssl/evp.h>
#endif

/* Linux x86-64. The trailer holds a gzip payload offset and a SHA-256 build ID. */
static char *staging;

static void die(const char *msg) {
    perror(msg);
    exit(1);
}

static void invalid(const char *msg) {
    fprintf(stderr, "Invalid training release: %s\n", msg);
    exit(1);
}

static char *join(const char *a, const char *b) {
    char *path;
    if (asprintf(&path, "%s/%s", a, b) < 0) die("asprintf");
    return path;
}

static int discard(const char *path, const struct stat *st, int type, struct FTW *state) {
    (void)st; (void)type; (void)state;
    return remove(path);
}

static void clean_stage(void) {
    if (staging) nftw(staging, discard, 24, FTW_DEPTH | FTW_PHYS);
}

static void gz_exact(gzFile input, void *buffer, unsigned size) {
    unsigned char *p = buffer;
    while (size) {
        int n = gzread(input, p, size);
        if (n <= 0) invalid("truncated or corrupt compressed payload");
        p += n;
        size -= (unsigned)n;
    }
}

static void write_exact(int fd, const unsigned char *data, size_t size, const char *path) {
    while (size) {
        ssize_t n = write(fd, data, size);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) die(path);
        data += n;
        size -= (size_t)n;
    }
}

#ifdef RLINF_SEALED_MODULES
static void extract_encrypted(gzFile input, uint64_t size, const char *name,
                              const char *path, unsigned mode) {
    const unsigned nonce_size = 12, tag_size = 16;
    if (size <= nonce_size + tag_size || size > (256u << 20))
        invalid("invalid encrypted module size");
    unsigned char nonce[12], tag[16];
    int length = (int)(size - nonce_size - tag_size), written = 0, final = 0;
    unsigned char *data = malloc((size_t)length + EVP_MAX_BLOCK_LENGTH);
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!data || !ctx) die("allocating native module decoder");
    gz_exact(input, nonce, nonce_size);
    gz_exact(input, data, (unsigned)length);
    gz_exact(input, tag, tag_size);
    /* GCM permits in-place decryption. Publish only authenticated bytes. */
    if (EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, rlinf_key_record + 16, nonce) != 1 ||
        EVP_DecryptUpdate(ctx, NULL, &written, (const unsigned char *)name, (int)strlen(name)) != 1 ||
        EVP_DecryptUpdate(ctx, data, &written, data, length) != 1 || written != length ||
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, tag_size, tag) != 1 ||
        EVP_DecryptFinal_ex(ctx, data + written, &final) != 1 || final != 0)
        invalid("encrypted module authentication failed");
    EVP_CIPHER_CTX_free(ctx);
    if (length < 4 || memcmp(data, "\x7f" "ELF", 4)) invalid("expected a native module");
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, mode & 0777);
    if (fd < 0) die(path);
    write_exact(fd, data, (size_t)length, path);
    if (close(fd)) die(path);
    free(data);
}
#endif

static char *read_string(gzFile input, unsigned size) {
    if (!size || size >= PATH_MAX) invalid("invalid path length");
    char *s = malloc((size_t)size + 1);
    if (!s) die("malloc");
    gz_exact(input, s, size);
    if (memchr(s, '\0', size)) invalid("NUL in pathname");
    s[size] = 0;
    return s;
}

static void validate_name(const char *s) {
    for (const char *p = s;;) {
        size_t n = strcspn(p, "/");
        if (!n || (n == 1 && *p == '.') || (n == 2 && !strncmp(p, "..", 2)))
            invalid("non-relative member");
        if (!p[n]) break;
        p += n + 1;
    }
}

static void extract(int fd, uint64_t offset, const char *cache, const char *dest) {
    if (lseek(fd, (off_t)offset, SEEK_SET) < 0) die("seek");
    gzFile input = gzdopen(fd, "rb");
    if (!input) die("gzdopen");
    gzbuffer(input, 1 << 18);
    staging = join(cache, ".unpack-XXXXXX");
    if (!mkdtemp(staging)) die("creating runtime staging directory");
    if (atexit(clean_stage)) die("atexit");
    unsigned char data[1 << 16];
    for (;;) {
        unsigned char kind;
        gz_exact(input, &kind, 1);
        if (!kind) break;
        uint16_t mode;
        uint32_t namesize;
        uint64_t size;
        gz_exact(input, &mode, sizeof(mode));
        gz_exact(input, &namesize, sizeof(namesize));
        gz_exact(input, &size, sizeof(size));
        char *name = read_string(input, namesize);
        validate_name(name);
        char *path = join(staging, name);
        if (kind == 'D' && size == 0) {
            if (mkdir(path, 0755)) die(path);
        } else if (kind == 'F') {
            int file = open(path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, mode & 0777);
            if (file < 0) die(path);
            while (size) {
                unsigned n = size < sizeof(data) ? (unsigned)size : sizeof(data);
                gz_exact(input, data, n);
                write_exact(file, data, n, path);
                size -= n;
            }
            if (close(file)) die(path);
#ifdef RLINF_SEALED_MODULES
        } else if (kind == 'E') {
            extract_encrypted(input, size, name, path, mode);
#endif
        } else if (kind == 'L' && size < PATH_MAX) {
            char *target = read_string(input, (unsigned)size);
            if (target[0] == '/') invalid("absolute symlink");
            if (symlink(target, path)) die(path);
            free(target);
        } else invalid("unknown member type");
        free(name);
        free(path);
    }
    if (gzread(input, data, 1) != 0 || !gzeof(input) || gzclose(input) != Z_OK)
        invalid("payload checksum or end of archive");
    if (rename(staging, dest)) die("publish extracted environment");
    free(staging);
    staging = NULL;
}

/* The optional gzip tar ends immediately before this descriptor. The main
 * application starts immediately after it. Neither query extracts a file. */
static int offline_runtime(int fd, uint64_t app_offset, int print_hash) {
    unsigned char trailer[56], buffer[1 << 16];
    if (app_offset < sizeof(trailer)) return 3;
    uint64_t end = app_offset - sizeof(trailer), start, size;
    if (pread(fd, trailer, sizeof(trailer), (off_t)end) != sizeof(trailer))
        invalid("cannot read offline runtime descriptor");
    if (memcmp(trailer, "RLOFFL01", 8)) return 3;
    memcpy(&start, trailer + 8, sizeof(start));
    memcpy(&size, trailer + 16, sizeof(size));
    if (!start || start >= end || size != end - start) invalid("invalid offline runtime bounds");
    if (print_hash) {
        for (int i = 24; i < 56; ++i) printf("%02x", trailer[i]);
        return puts("") < 0 ? 1 : 0;
    }
    if (lseek(fd, (off_t)start, SEEK_SET) < 0) die("seeking offline runtime");
    while (size) {
        size_t wanted = size < sizeof(buffer) ? (size_t)size : sizeof(buffer);
        ssize_t n = read(fd, buffer, wanted);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) invalid("truncated offline runtime");
        for (ssize_t pos = 0; pos < n;) {
            ssize_t out = write(STDOUT_FILENO, buffer + pos, (size_t)(n - pos));
            if (out < 0 && errno == EINTR) continue;
            if (out <= 0) die("streaming offline runtime");
            pos += out;
        }
        size -= (uint64_t)n;
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *usage = "Usage: ./run COMMAND CONFIG.yaml [Hydra arguments]\n"
                        "       ./run COMMAND REGISTERED_CONFIG [Hydra arguments]\n"
                        "Example: ./run embodiment example.yaml\n";
    if (argc >= 2 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) {
        fputs(usage, stdout);
        puts("Supported commands:\n" RLINF_SCRIPT_LIST);
        return 0;
    }
    int offline_hash = argc == 2 && !strcmp(argv[1], "--offline-runtime-sha256");
    int offline_tar = argc == 2 && !strcmp(argv[1], "--offline-runtime-archive");
    if (!offline_hash && !offline_tar && !(argc == 2 && !strcmp(argv[1], "--runtime-dir")) &&
        (argc < 3 || argv[2][0] == '-')) {
        fputs(usage, stderr);
        fputs("Supply the command first and its configuration second. See --help for command names.\n", stderr);
        return 2;
    }
    char *exe = realpath("/proc/self/exe", NULL);
    if (!exe) die("locating executable");
    int fd = open(exe, O_RDONLY | O_CLOEXEC);
    struct stat st;
    unsigned char footer[48];
    if (fd < 0 || fstat(fd, &st)) die("opening release");
    if (st.st_size < (off_t)sizeof(footer) ||
        pread(fd, footer, sizeof(footer), st.st_size - sizeof(footer)) != sizeof(footer) ||
        memcmp(footer, "RLFRUN01", 8)) invalid("missing runtime trailer");
    uint64_t offset;
    memcpy(&offset, footer + 8, sizeof(offset));
    if (offset < 1 || offset >= (uint64_t)st.st_size - sizeof(footer)) invalid("invalid offset");
    if (offline_hash || offline_tar) return offline_runtime(fd, offset, offline_hash);
    char id[65];
    for (int i = 0; i < 32; i++) snprintf(id + 2*i, 3, "%02x", footer[16+i]);
    char *slash = strrchr(exe, '/');
    if (!slash) die("locating release directory");
    *slash = 0;
    const char *root = *exe ? exe : "/";
    char *cache = join(root, ".train-app");
    if (mkdir(cache, 0700) && errno != EEXIST) die(cache);
    char *dest = join(cache, id);
    int lock = open(join(cache, ".lock"), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (lock < 0 || flock(lock, LOCK_EX)) die("locking local runtime");
    if (stat(dest, &st)) {
        if (errno != ENOENT) die(dest);
        fprintf(stderr, "Unpacking compiled training modules beside the executable...\n");
        extract(fd, offset, cache, dest);
    } else {
        if (!S_ISDIR(st.st_mode)) invalid("runtime path is not a directory");
        close(fd);
    }
    close(lock);
    if (argc == 2 && !strcmp(argv[1], "--runtime-dir")) {
        puts(dest);
        return 0;
    }
    char *env = realpath(join(root, ".venv"), NULL);
    char *python = env ? join(env, "bin/python") : NULL;
    if (!python || access(python, X_OK)) {
        /* Use the bundled seed to diagnose the requested script/config before setup. */
        env = join(dest, "setup/seed");
        python = join(env, "bin/python");
        if (access(python, X_OK)) {
            fputs("Install the embedded task environment with ./install.sh --offline first.\n", stderr);
            return 2;
        }
    }
    char *libs = join(env, "lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib");
    char *binpath = join(env, "bin:/usr/bin:/bin:/usr/sbin:/sbin");
    if (setenv("LD_LIBRARY_PATH", libs, 1) || setenv("PATH", binpath, 1) ||
        setenv("TRAIN_RELEASE_ROOT", root, 1)) die("setting bundled environment");
    unsetenv("LD_PRELOAD");
    unsetenv("LD_AUDIT");
    unsetenv("PYTHONPATH");
    unsetenv("PYTHONHOME");
    unsetenv("VIRTUAL_ENV");
    char **args = calloc((size_t)argc + 5, sizeof(*args));
    if (!args) die("calloc");
    args[0] = python;
    args[1] = "-I";
    args[2] = "-c";
    args[3] = "import sys; sys.path.insert(0,sys.argv.pop(1)); "
              "from _train_release_start import main; sys.exit(main())";
    args[4] = dest;
    for (int i = 1; i < argc; i++) args[i + 4] = argv[i];
    execv(python, args);
    die(python);
}
