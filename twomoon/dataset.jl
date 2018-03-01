using Plots;

function generateTwoMoonDS(count; T::Type = Float32, std = 0.025f0)::AbstractVector
	ds = Vector{Tuple{Vector, Int}}(count);
	for i in 1:length(ds)
		label = rand(0:1);
		if label == 0
			x = 0.3f0 + rand(T) * 0.6f0;
			feature = [x, -2*sqrt(0.3^2 - (x - 0.6)^2) + 0.7] .+ (randn(2) .* std);
		else
			x = 0.1f0 + rand(T) * 0.6f0;
			feature = [x, 2*sqrt(0.3^2 - (x - 0.4)^2) + 0.3] .+ (randn(2) .* std);
		end
		ds[i] = (feature, label)
	end
	return ds;
end

function plotTwoMoonDS(ds::AbstractVector)
	negds = getindex.(ds[getindex.(ds, 2) .== 0], 1);
	posds = getindex.(ds[getindex.(ds, 2) .== 1], 1);
	scatter(getindex.(negds, 1), getindex.(negds, 2), label = "negative", aspect_ratio = :equal, xlims=(0:1), ylims=(0:1));
	scatter!(getindex.(posds, 1), getindex.(posds, 2), label = "positive");
end
