using Flux;
using Plots;

function generateTwoMoonDS(count::Int; T::Type = Float32, std = 0.025f0)::Tuple{AbstractMatrix, Flux.OneHotMatrix}
	X = Matrix{T}(undef, 2, count);
	Y = Vector{Int}(undef, count);
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
	res = scatter(getindex(negds, 1, :), getindex(negds, 2, :), label = "negative", aspect_ratio = :equal, xlims = (0:1), ylims = (0:1));
	scatter!(res, getindex(posds, 1, :), getindex(posds, 2, :), label = "positive");
	return res;
end
