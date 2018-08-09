using Plots;

function generateTwoMoonDS(count::Int; T::Type = Float32, std = 0.025f0)::Tuple{AbstractMatrix, AbstractVector}
	X = Matrix{T}(2, count);
	Y = Vector{Int}(count);
	for i in 1:count
		Y[i] = rand(1:2);
		if Y[i] == 1
			x = 0.3f0 + rand(T) * 0.6f0;
			X[:, i] = [x, -2*sqrt(0.3^2 - (x - 0.6)^2) + 0.7] .+ (randn(2) .* std);
		else
			x = 0.1f0 + rand(T) * 0.6f0;
			X[:, i] = [x, 2*sqrt(0.3^2 - (x - 0.4)^2) + 0.3] .+ (randn(2) .* std);
		end
	end
	return (X, Y);
end

function plotTwoMoonDS(ds::Tuple{AbstractMatrix, AbstractVector})
	(X, Y) = ds
	negds = X[:, Y .== 1];
	posds = X[:, Y .== 2];
	res = scatter(getindex(negds, 1, :), getindex(negds, 2, :), label = "negative", aspect_ratio = :equal, xlims = (0:1), ylims = (0:1));
	scatter!(res, getindex(posds, 1, :), getindex(posds, 2, :), label = "positive");
	return res;
end
