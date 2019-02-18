#include <stdio.h>
#include <stdarg.h>

#define N (125)

#define SNPRINTF_FOR_NORMAL_PEOPLE(buf, n, format, ...) \
  snprintf(buf, n, ":parrot: " format, __VA_ARGS__)

#define SNPRINTF_FOR_MIC_SAN(buf, n, format, ...) ({ \
  struct struct_for_snprintf_function { \
    static int function_for_snprintf_calling( \
        char *_buf, size_t _n, const char *_format, ...) { \
      va_list args; \
      va_start(args, _format); \
      int _retinmacro = vsnprintf(_buf, _n, _format, args); \
      va_end(args); \
      return _retinmacro; \
    } \
  }; \
  struct_for_snprintf_function::function_for_snprintf_calling( \
    buf, n, ":parrot: " format, __VA_ARGS__ \
  ); \
})

int main(void) {
  char buf[N+3] = {0};  /* 3 as margin */
  int ret = 0;
  
  ret = SNPRINTF_FOR_NORMAL_PEOPLE(buf, N, "Hello %s-san!!", "Shimodaira");
  printf("My  result[%d]=> %s\n", ret, buf);

  ret = SNPRINTF_FOR_MIC_SAN(buf, N, "Hello %s-san!!", "Tanaka");
  printf("Mic result[%d]=> %s\n", ret, buf);

  ret = SNPRINTF_FOR_MIC_SAN(buf, N, "Hello %s-san!!", "Sato");
  printf("Mic result[%d]=> %s\n", ret, buf);
  return 0;
}
