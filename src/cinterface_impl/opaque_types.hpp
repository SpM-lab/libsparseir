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
    int spir_is_assigned_##name(const spir_##name *obj)                        \
    {                                                                          \
        if (!obj) {                                                            \
            DEBUG_LOG(#name << " object is null");                             \
            return 0;                                                          \
        }                                                                      \
        bool is_assigned = static_cast<bool>(obj->ptr);                        \
        DEBUG_LOG(#name << " object at " << obj << ", ptr=" << obj->ptr.get()  \
                        << ", is_assigned=" << is_assigned);                   \
        return is_assigned ? 1 : 0;                                            \
    }                                                                          \
                                                                               \
    /* Clone function */                                                       \
    spir_##name *spir_clone_##name(const spir_##name *src)                     \
    {                                                                          \
        DEBUG_LOG("Cloning " << #name << " at " << src);                       \
        if (!src) {                                                            \
            DEBUG_LOG("Source " << #name << " is null");                       \
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
                DEBUG_LOG("Cloned " << #name << " to " << result               \
                                    << ", shared_ptr points to "               \
                                    << result->ptr.get());                     \
            } else {                                                           \
                DEBUG_LOG("Source " << #name << " has null shared_ptr");       \
                result->ptr = nullptr;                                         \
            }                                                                  \
                                                                               \
            return result;                                                     \
        } catch (const std::exception &e) {                                    \
            DEBUG_LOG("Exception in " << #name << "_clone: " << e.what());     \
            return nullptr;                                                    \
        } catch (...) {                                                        \
            DEBUG_LOG("Unknown exception in " << #name << "_clone");           \
            return nullptr;                                                    \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* Destroy function */                                                     \
    void spir_destroy_##name(spir_##name *obj)                                 \
    {                                                                          \
        if (!obj) {                                                            \
            DEBUG_LOG(#name << " object is null");                             \
            return;                                                            \
        }                                                                      \
        DEBUG_LOG("Destroying " << #name << " object at " << obj);             \
        /* Check before resetting */                                           \
        if (obj->ptr) {                                                        \
            DEBUG_LOG("Resetting shared_ptr in " << #name << " at "            \
                                                 << obj->ptr.get());           \
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
            DEBUG_LOG(#name << " object is null");                             \
            return nullptr;                                                    \
        }                                                                      \
        DEBUG_LOG(#name << " object at " << obj                                \
                        << ", ptr=" << obj->ptr.get());                        \
        return obj->ptr;                                                       \
    }

IMPLEMENT_OPAQUE_TYPE(kernel, sparseir::AbstractKernel);
IMPLEMENT_OPAQUE_TYPE(logistic_kernel, sparseir::LogisticKernel);
IMPLEMENT_OPAQUE_TYPE(regularized_bose_kernel, sparseir::RegularizedBoseKernel);
IMPLEMENT_OPAQUE_TYPE(singular_funcs, AbstractContinuousFunctions);
IMPLEMENT_OPAQUE_TYPE(matsubara_functions, AbstractMatsubaraFunctions);
IMPLEMENT_OPAQUE_TYPE(fermionic_finite_temp_basis,
                      sparseir::FiniteTempBasis<sparseir::Fermionic>);
IMPLEMENT_OPAQUE_TYPE(bosonic_finite_temp_basis,
                      sparseir::FiniteTempBasis<sparseir::Bosonic>);
IMPLEMENT_OPAQUE_TYPE(sampling, sparseir::AbstractSampling);
IMPLEMENT_OPAQUE_TYPE(sve_result, sparseir::SVEResult);
IMPLEMENT_OPAQUE_TYPE(
    fermionic_dlr,
    sparseir::DiscreteLehmannRepresentation<sparseir::Fermionic>);
IMPLEMENT_OPAQUE_TYPE(
    bosonic_dlr, sparseir::DiscreteLehmannRepresentation<sparseir::Bosonic>);