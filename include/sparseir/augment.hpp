#pragma once

#include <memory>
#include <vector>
#include <Eigen/Dense>

namespace sparseir{

    template <typename FB, typename FA>
    struct AbstractAugmentedFunction
    {
        FB fbasis;
        FA faug;

        // Default constructor
        AbstractAugmentedFunction() = default;

        Eigen::VectorXd operator()(const Eigen::VectorXd &x) const
        {
            Eigen::VectorXd fbasis_x = fbasis(x);
            Eigen::VectorXd faug_x(faug.size());
            for (size_t i = 0; i < faug.size(); ++i) {
                faug_x[i] = faug[i](x[i]);
            }
            Eigen::VectorXd result(faug_x.size() + fbasis_x.size());
            result << faug_x, fbasis_x; // Concatenate vectors
            return result;
        }

        Eigen::VectorXd operator()(double x) const
        {
            Eigen::VectorXd fbasis_x = fbasis(Eigen::VectorXd::Constant(1, x));
            Eigen::VectorXd faug_x(faug.size());
            for (size_t i = 0; i < faug.size(); ++i) {
                faug_x[i] = faug[i](x);
            }
            Eigen::VectorXd result(faug_x.size() + fbasis_x.size());
            result << faug_x, fbasis_x; // Concatenate vectors
            return result;
        }

        // Virtual function using base type pointers
        virtual AbstractAugmentedFunction<FB, FA> augmentedfunction()
        {
            // Implement functionality here
            return *this;
        }

        virtual FB get_fbasis(AbstractAugmentedFunction a) {
            return a.augmented_basis.fbasis;
        }
        virtual FA get_abasis(AbstractAugmentedFunction a) {
            return a.augmented_basis.faug;
        }


    };

    // Make AugmentedFunction a template class
    template <typename FB, typename FA>
    struct AugmentedFunction : AbstractAugmentedFunction<FB, FA>
    {
        // Override augmentedfunction
        AugmentedFunction<FB, FA> augmentedfunction() override { return *this; }
    };

    /*
    template <typename FB, typename FA>
    struct AugmentedTauFunction : AbstractAugmentedFunction{

    }
    */

    /*
    template <typename S, typename B, typename F, typename FT>
    struct AugmentedBasis: AbstractBasis<S> {
        S basis;
        std::vector<AbstractAugmentedFunction> augmentations;
        F u;
        FT uhat;

        // Constructor
        template <typename A>
        AugmentedBasis(
            AbstractBasis<S> basis,
            std::vector<A> augmentations)
        {
            basis->basis;
            this-> augmentations = augmentations;

        }
    };
    */
} // namespace sparseir