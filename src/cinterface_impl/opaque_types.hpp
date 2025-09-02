#include <memory>
#include <iostream>

// Define opaque type and implement its management functions
#define IMPLEMENT_OPAQUE_TYPE(name, impl_type)                                 \
    struct _spir_##name                                                        \
    {                                                                          \
        std::shared_ptr<impl_type> ptr;                                        \
        _spir_##name() { }                                                     \
        ~_spir_##name() { }                                                    \
    };                                                                         \
    typedef struct _spir_##name spir_##name;                                   \
                                                                               \
    /* Helper function for creating objects */                                 \
    static inline spir_##name *create_##name(std::shared_ptr<impl_type> p)     \
    {                                                                          \
        auto *obj = new spir_##name;                                           \
        obj->ptr = p;                                                          \
        return obj;                                                            \
    }                                                                          \
                                                                               \
    /* Check if the shared_ptr has a valid object */                           \
    int spir_##name##_is_assigned(const spir_##name *obj)                        \
    {                                                                          \
        if (!obj) {                                                            \
            DEBUG_LOG(std::string(#name) + " object is null");                \
            return 0;                                                          \
        }                                                                      \
        bool is_assigned = static_cast<bool>(obj->ptr);                        \
        DEBUG_LOG(std::string(#name) + " object at " + std::to_string(reinterpret_cast<uintptr_t>(obj)) + \
                  ", std::shared_ptr at " + std::to_string(reinterpret_cast<uintptr_t>(obj->ptr.get())) + \
                  ", raw ptr=" + std::to_string(reinterpret_cast<uintptr_t>(obj->ptr.get())) + \
                  ", use_count=" + std::to_string(obj->ptr.use_count()) + \
                  ", is_assigned=" + (is_assigned ? "true" : "false"));        \
        if (is_assigned) {                                                    \
            if (obj->ptr.use_count() == 0) {                                  \
                DEBUG_LOG(std::string(#name) + " object has 0 use_count");    \
                return 0;                                                     \
            }                                                                  \
        }                                                                      \
        return is_assigned ? 1 : 0;                                            \
    }                                                                          \
                                                                               \
    /* Clone function */                                                       \
    spir_##name *spir_##name##_clone(const spir_##name *src)                     \
    {                                                                          \
        DEBUG_LOG("Cloning " + std::string(#name) + " at " + std::to_string(reinterpret_cast<uintptr_t>(src))); \
        if (!src) {                                                            \
            DEBUG_LOG("Source " + std::string(#name) + " is null");           \
            return nullptr;                                                    \
        }                                                                      \
                                                                               \
        try {                                                                  \
            /* Create a new structure */                                       \
            spir_##name *result = new spir_##name();                           \
                                                                               \
            /* If source has a valid shared_ptr, copy it */                    \
            if (src->ptr) {                                                    \
                /* Create a new shared_ptr instance that shares ownership */   \
                result->ptr = src->ptr;                                        \
                DEBUG_LOG("Cloned " + std::string(#name) + " to " + std::to_string(reinterpret_cast<uintptr_t>(result)) + \
                          ", shared_ptr points to " + std::to_string(reinterpret_cast<uintptr_t>(result->ptr.get())));                     \
            } else {                                                           \
                DEBUG_LOG("Source " + std::string(#name) + " has null shared_ptr");       \
                result->ptr = nullptr;                                         \
            }                                                                  \
                                                                               \
            return result;                                                     \
        } catch (const std::exception &e) {                                    \
            DEBUG_LOG("Exception in " + std::string(#name) + "_clone: " + e.what());     \
            return nullptr;                                                    \
        } catch (...) {                                                        \
            DEBUG_LOG("Unknown exception in " + std::string(#name) + "_clone");           \
            return nullptr;                                                    \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* Destroy function */                                                     \
    void spir_##name##_release(spir_##name *obj)                                 \
    {                                                                          \
        if (!obj) {                                                            \
            DEBUG_LOG(std::string(#name) + " object is null");                \
            return;                                                            \
        }                                                                      \
        DEBUG_LOG("Destroying " + std::string(#name) + " object at " + std::to_string(reinterpret_cast<uintptr_t>(obj))); \
        /* Check before resetting */                                           \
        if (obj->ptr) {                                                        \
            DEBUG_LOG("Resetting shared_ptr in " + std::string(#name) + " at " + \
                      std::to_string(reinterpret_cast<uintptr_t>(obj->ptr.get()))); \
            obj->ptr.reset();                                                  \
        }                                                                      \
        /* Safely delete the object */                                         \
        delete obj;                                                            \
    }                                                                          \
                                                                               \
    /* Helper to get the implementation shared_ptr */                          \
    static inline std::shared_ptr<impl_type> get_impl_##name(                  \
        const spir_##name *obj)                                                \
    {                                                                          \
        if (!obj) {                                                            \
            DEBUG_LOG(std::string(#name) + " object is null");                \
            return nullptr;                                                    \
        }                                                                      \
        DEBUG_LOG(std::string(#name) + " object at " + std::to_string(reinterpret_cast<uintptr_t>(obj)) + \
                  ", ptr=" + std::to_string(reinterpret_cast<uintptr_t>(obj->ptr.get()))); \
        if (!obj->ptr) {                                                       \
            DEBUG_LOG(std::string(#name) + " object has null shared_ptr");    \
            return nullptr;                                                    \
        }                                                                      \
        return obj->ptr;                                                       \
    }

IMPLEMENT_OPAQUE_TYPE(kernel, sparseir::AbstractKernel);
IMPLEMENT_OPAQUE_TYPE(funcs, _AbstractFuncs);
IMPLEMENT_OPAQUE_TYPE(basis, AbstractFiniteTempBasis);
IMPLEMENT_OPAQUE_TYPE(sampling, sparseir::AbstractSampling);
IMPLEMENT_OPAQUE_TYPE(sve_result, sparseir::SVEResult);
IMPLEMENT_OPAQUE_TYPE(dlr, AbstractFiniteTempBasis);