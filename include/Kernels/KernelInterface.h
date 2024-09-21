#ifndef KERNEL_INTERFACE_H
#define KERNEL_INTERFACE_H

#include <string>

class KernelInterface{
    public:
        virtual ~KernelInterface(){};
        virtual void initialize() = 0;
        virtual void shutdown() = 0;
        virtual void saveStats(const std::string &file_path) = 0;
        virtual void launch() = 0;
};

#endif
