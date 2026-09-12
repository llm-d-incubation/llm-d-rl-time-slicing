#ifndef BACKEND_H
#define BACKEND_H

#include <mutex>

class Backend {
public:
    Backend(int id);
    virtual ~Backend();
    virtual void setup();
    virtual void* get_tmp_buf() = 0;
    virtual void* get_host_buffer() = 0;
};

class ShareMem : public Backend {
public:
    void *tmp_buf = nullptr;
    void* host_buf_ptr = nullptr;
    int fd_host = -1;

    // Guards tmp_buf creation/publication only (setup(), deferred-mode
    // get_tmp_buf()). Distinct from vGPU.cpp's global fs_mutex, which is
    // the op-scope lock handlers hold across whole checkpoint/restore
    // operations — the two must never be conflated.
    std::mutex buf_mutex;

    int id;

    ShareMem(int id);
    ~ShareMem();
    void setup() override;
    void* get_tmp_buf() override;
    void* get_host_buffer();

private:
    void* map_dump_buffer(bool fatal);
};

#endif