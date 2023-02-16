#pragma once

#include <assert.h>

#include <stdarg.h>
#include <stdint.h>
#include <stddef.h>

#include <stdlib.h>
#include <stdio.h>

typedef struct fmt_buf
{
    size_t len, capacity;
    char *buf;
} fmt_buf;

static inline fmt_buf make_buf()
{
    fmt_buf res;
    res.len = 0;
    res.capacity = 0;
    res.buf = NULL;
    return res;
}

static inline void free_buf(fmt_buf *buf)
{
    free(buf->buf);
    buf->capacity = 0;
    buf->len = 0;
}

static inline int buf_printf(fmt_buf *buf, const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);

    size_t remaining = buf->capacity - buf->len;
    int ret = vsnprintf(buf->buf + buf->len, remaining, fmt, args);
    if (ret < 0)
        return ret;

    if ((unsigned int) ret + 1 > remaining)
    {
        size_t new_cap = buf->capacity * 1.5;
        if (new_cap < buf->capacity + ret + 1)
        {
            new_cap = buf->capacity + ret + 1;
        }

        char *new_buf = (char *) realloc(buf->buf, new_cap);
        if (new_buf == NULL)
            return -1;
        buf->buf = new_buf;
        buf->capacity = new_cap;

        remaining = buf->capacity - buf->len;
        ret = vsnprintf(buf->buf + buf->len, remaining, fmt, args);
        assert(ret > 0 && ret < remaining);
    }

    buf->len += ret;

    va_end(args);
    return ret;
}
