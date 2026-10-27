#include <cstdio>

int vnpu_run_selftest(const char *filter);

int main(int argc, char **argv)
{
    const char *filter = argc > 1 ? argv[1] : nullptr;
    return vnpu_run_selftest(filter);
}
