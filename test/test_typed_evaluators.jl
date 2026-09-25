using Test
using Oscar
using NonArchimedeanMachineLearning

@testset "Typed evaluators with wrapped coordinates" begin
    K = PadicField(2, 20)
    R, (x, y) = polynomial_ring(K, ["x", "y"])
    VFP = ValuedFieldPoint{2, 20, PadicFieldElem}
    PT = ValuationPolydisc{VFP, Int, 2}
    linear = LinearPolynomial(VFP.([K(3), K(2)]), VFP(K(1)))
    second = LinearPolynomial(VFP.([K(1), K(1)]), VFP(K(2)))
    p = ValuationPolydisc([K(1), K(2)], [3, 3])

    # Cancellation makes the linear value at the center 8, with disc norm 1/8.
    # Both native wrapped coefficients and adapters for raw coefficients are exercised.
    for (f, expected) in [(linear, 1 / 8),
            (NonArchimedeanMachineLearning.Constant{PadicFieldElem}(5.0), 5.0),
            (x^2 + y^2, 1.0),
            (LinearAbsolutePolynomialSum([linear, second]), 9 / 8)]
        @test batch_evaluate_init(f, PT)(p) == expected
    end

    @testset "Lifted polynomial evaluator" begin
        poly = x^3 + K(2) * x * y + K(1)
        lifted = batch_evaluate_init(poly, PT)
        @test lifted isa LiftedMPolyEvaluator
        raw_p = ValuationPolydisc{PadicFieldElem, Int, 2}((K(1), K(2)), (3, 3))
        @test lifted(p) == NonArchimedeanMachineLearning.evaluate(poly, raw_p)

        child = ValuationPolydisc([K(1), K(2)], [4, 3])
        raw_child = ValuationPolydisc{PadicFieldElem, Int, 2}((K(1), K(2)), (4, 3))
        tangent = ValuationTangent(p, child, zeros(Int, 2))
        raw_tangent = ValuationTangent(raw_p, raw_child, zeros(Int, 2))
        @test directional_derivative(lifted, tangent) ==
              directional_derivative(poly, raw_tangent)
    end
end
