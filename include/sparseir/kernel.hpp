// kernel.hpp
// Header file for kernels
// Ported from src/kernel.jl

#include <stdexcept>
#include <utility>
#include <functional>

namespace sparseir
{

    // Abstract base class for kernels
    class AbstractKernel
    {
    public:
        virtual ~AbstractKernel() {}

        // Evaluate the kernel at point (x, y)
        virtual double operator()(double x, double y) const = 0;

        // Return the range of x and y
        virtual std::pair<double, double> xrange() const { return {-1.0, 1.0}; }
        virtual std::pair<double, double> yrange() const { return {-1.0, 1.0}; }

        // Virtual method to check if the kernel is centrosymmetric
        virtual bool is_centrosymmetric() const
        {
            return false;
        }

        // Virtual method to get the symmetrized version of the kernel
        virtual std::shared_ptr<AbstractKernel> get_symmetrized(int sign)
        {
            // By default, throw an error if symmetrization is not implemented
            throw std::runtime_error("Symmetrization not implemented for this kernel type");
        }
    };

    // Derived kernel classes
    class LogisticKernel : public AbstractKernel
    {
    public:
        double Lambda;

        explicit LogisticKernel(double lambda) : Lambda(lambda) {}

        virtual double operator()(double x, double y) const override
        {
            // Provide implementation here
            // This is just a placeholder
            return x * Lambda + y;
        }

        // Override to indicate that LogisticKernel is centrosymmetric
        bool is_centrosymmetric() const override
        {
            return true;
        }

        // Override to provide symmetrization for LogisticKernel
        std::shared_ptr<AbstractKernel> get_symmetrized(int sign) override;

    };

    class RegularizedBoseKernel : public AbstractKernel
    {
    public:
        double Lambda;

        explicit RegularizedBoseKernel(double lambda) : Lambda(lambda) {}

        // Override to indicate that RegularizedBoseKernel is centrosymmetric
        bool is_centrosymmetric() const override
        {
            return true;
        }

        // Override to provide symmetrization for RegularizedBoseKernel
        std::shared_ptr<AbstractKernel> get_symmetrized(int sign) override;

        virtual double operator()(double x, double y) const override
        {
            // Provide implementation here
            // This is just a placeholder
            return x * Lambda + y;
        }
    };

// Abstract class for ReducedKernel
class AbstractReducedKernel : public AbstractKernel {
public:
    explicit AbstractReducedKernel(std::shared_ptr<AbstractKernel> inner_kernel, int sign)
        : inner_kernel_(inner_kernel), sign_(sign) {}

    // ReducedKernel is not centrosymmetric
    bool is_centrosymmetric() const override {
        return false;
    }

    // Cannot symmetrize a ReducedKernel again
    std::shared_ptr<AbstractKernel> get_symmetrized(int sign) override {
        throw std::runtime_error("Cannot symmetrize twice");
    }

    int get_sign() const {
        return sign_;
    }

    std::shared_ptr<AbstractKernel> get_inner_kernel() const {
        return inner_kernel_;
    }

protected:
    std::shared_ptr<AbstractKernel> inner_kernel_;
    int sign_;
};


    class ReducedKernel : public AbstractKernel
    {
    public:
        std::shared_ptr<AbstractKernel> innerKernel;
        int sign;

        ReducedKernel(std::shared_ptr<AbstractKernel> inner, int sign)
            : innerKernel(std::move(inner)), sign(sign) {}
    };


    // LogisticKernelOdd (for fermionic functions)
    class LogisticKernelOdd : public AbstractKernel
    {
    public:
        LogisticKernelOdd(std::shared_ptr<LogisticKernel> inner_kernel, int sign)
            : AbstractReducedKernel(inner_kernel, sign)
        {
            if (!inner_kernel->is_centrosymmetric())
            {
                throw std::runtime_error("Inner kernel must be centrosymmetric");
            }
            if (std::abs(sign) != 1)
            {
                throw std::domain_error("Sign must be -1 or 1");
            }
        }

        virtual double operator()(double x, double y) const override;

        virtual std::pair<double, double> xrange() const override;
        virtual std::pair<double, double> yrange() const override;

    private:
        const LogisticKernel &inner_kernel_;
        int sign_;
    };

    // RegularizedBoseKernelOdd (for bosonic functions)
    class RegularizedBoseKernelOdd : public AbstractKernel
    {
    public:
        RegularizedBoseKernelOdd(std::shared_ptr<RegularizedBoseKernel> inner_kernel, int sign)
            : AbstractReducedKernel(inner_kernel, sign)
        {
            if (!inner_kernel->is_centrosymmetric())
            {
                throw std::runtime_error("Inner kernel must be centrosymmetric");
            }
            if (std::abs(sign) != 1)
            {
                throw std::domain_error("Sign must be -1 or 1");
            }
        }

        virtual double operator()(double x, double y) const override;

        virtual std::pair<double, double> xrange() const override;
        virtual std::pair<double, double> yrange() const override;

    private:
        const RegularizedBoseKernel &inner_kernel_;
        int sign_;
    };

    // Implementation of get_symmetrized for RegularizedBoseKernel
    std::shared_ptr<AbstractKernel> RegularizedBoseKernel::get_symmetrized(int sign)
    {
        if (sign == -1)
        {
            // Return a RegularizedBoseKernelOdd instance
            return std::make_shared<RegularizedBoseKernelOdd>(
                std::make_shared<RegularizedBoseKernel>(*this), sign);
        }
        else
        {
            // Call base implementation for other signs
            return std::make_shared<AbstractReducedKernel>(
                std::make_shared<RegularizedBoseKernel>(*this), sign);
        }
    }

    // Additional functions
    double weight_func(const LogisticKernel &kernel, const std::string &statistics, double y);
    double weight_func(const RegularizedBoseKernel &kernel, const std::string &statistics, double y);

    // Base class for SVE hints
    class AbstractSVEHints
    {
    public:
        virtual ~AbstractSVEHints() {}
        // Add virtual functions if necessary
    };

    // Derived SVE hints classes
    class SVEHintsLogistic : public AbstractSVEHints
    {
    public:
        LogisticKernel kernel;
        double epsilon;

        SVEHintsLogistic(const LogisticKernel &kernel_, double epsilon_)
            : kernel(kernel_), epsilon(epsilon_) {}
    };

    class SVEHintsRegularizedBose : public AbstractSVEHints
    {
    public:
        RegularizedBoseKernel kernel;
        double epsilon;

        SVEHintsRegularizedBose(const RegularizedBoseKernel &kernel_, double epsilon_)
            : kernel(kernel_), epsilon(epsilon_) {}
    };

    class SVEHintsReduced : public AbstractSVEHints
    {
    public:
        std::shared_ptr<AbstractSVEHints> innerHints;

        explicit SVEHintsReduced(std::shared_ptr<AbstractSVEHints> innerHints_)
            : innerHints(std::move(innerHints_)) {}
    };

    // Overloaded functions for sve_hints
    std::shared_ptr<AbstractSVEHints> sve_hints(const LogisticKernel &kernel, double epsilon)
    {
        return std::make_shared<SVEHintsLogistic>(kernel, epsilon);
    }

    std::shared_ptr<AbstractSVEHints> sve_hints(const RegularizedBoseKernel &kernel, double epsilon)
    {
        return std::make_shared<SVEHintsRegularizedBose>(kernel, epsilon);
    }

    /*
    std::shared_ptr<AbstractSVEHints> sve_hints(const ReducedKernel &kernel, double epsilon)
    {
        auto inner_hints = sve_hints(*kernel.innerKernel, epsilon);
        return std::make_shared<SVEHintsReduced>(inner_hints);
    }
    */

} // namespace sparseir
