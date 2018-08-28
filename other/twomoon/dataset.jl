using Flux;
using PGFPlots;

function generateTwoMoonDS(count::Int; T::Type = Float32, std = 0.025f0)::Tuple{AbstractMatrix, Flux.OneHotMatrix}
	X = Matrix{T}(2, count);
	Y = Vector{Int}(count);
	for i in 1:count
		Y[i] = rand(1:2);
		if Y[i] == 1
			x = 0.3f0 + rand(T) * 0.6f0;
			X[:, i] = [x, -2*sqrt(0.3^2 - (x - 0.6)^2) + 0.7] .+ (randn(2) .* std);
		else
			x = 0.1f0 + rand(T) * 0.6f0;
			X[:, i] = [x, +2*sqrt(0.3^2 - (x - 0.4)^2) + 0.3] .+ (randn(2) .* std);
		end
	end
	return (X, Flux.onehotbatch(Y, 1:2));
end

function plotTwoMoonDS(ds::Tuple{AbstractMatrix, Flux.OneHotMatrix})
	(X, Y) = ds;
	Y = Y[2, :];
	negds = X[:, Y .== false];
	posds = X[:, Y .== true];
	res = Axis([Plots.Scatter(getindex(negds, 1, :), getindex(negds, 2, :), legendentry = "negative"), Plots.Scatter(getindex(posds, 1, :), getindex(posds, 2, :), legendentry = "positive")], axisEqual = true, xmin = 0, xmax = 1, ymin = 0, ymax = 1);
	return res;
end
