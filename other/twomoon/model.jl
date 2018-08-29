using Flux;
using PGFPlots;
using Colors;

include("dataset.jl")

function plotModel(ds::Tuple{AbstractMatrix, Flux.OneHotMatrix}, model, performance)
	plots = [plotTwoMoonDS(testDS)];
	push!(plots[1], Plots.Image((x,y)->Flux.data(softmax(model([y, x])))[2], (0,1), (0, 1), colormap = ColorMaps.RGBArrayMap(colormap("RdBu"), invert = true, interpolation_levels= 500), zmin = 0, zmax = 1));
	#push!(plots, plot(performance, label = "Loss", ylims = (0.001, 1), yscale = :log10, title = "Final loss"));
	i = 1;
	jacobian = Vector{Matrix}();
	for layer in model
		if isa(layer, Target)
			for (key, value) in layer.debuglog
				if contains(key, "angle")
					#push!(plots, plot(value, label = "", title = "Layer " * string(i) * ": " * key, ylims = (0,180),))
				elseif contains(key, "loss")
					#push!(plots, plot(value, label = "", title = "Layer " * string(i) * ": " * key, ylims = (0.001,1), yscale = :log10))
				elseif contains(key, "jacobian")
					if size(jacobian) == (0,)
						jacobian = value;
					else
						jacobian = value .* jacobian;
					end
				else
					push!(plots, plot(value, label = "", title = "Layer " * string(i) * ": " * key, ylims = (0, 1)))
				end
			end
		end
		i += 1
	end
	if size(jacobian) != (0,)
		singularvalues = map(x->svdvals(x'), jacobian);
		ratios = map(x->x[1] / x[end], singularvalues);
		#push!(plots, plot(acosd.(1 ./ ratios), label = "", title = "Maximum angle by theorem", ylims = (-10,180),))
	end
	return plots;
end

numBatches = 2000;
σ = Flux.relu;
noiseDeviation = 0.2;
loss = Flux.mse;
modelloss(x, y) = Flux.mse(softmax(x), y);

Layer(in::Int, out::Int) = Target(Chain(Dense(in, out, σ)), Chain(Dense(out, 8, σ), Dense(8, in, identity)), loss; σ = noiseDeviation);

model = Chain(Layer(2, 16), Layer(16, 2));
trainDS = generateTwoMoonDS(1000);
testDS = generateTwoMoonDS(1000);

performance = Vector{Float32}(numBatches);

for i in 1:numBatches
	targettrain!(model, modelloss, [trainDS], ADAM(params(model)), η = 0.5, cb = () -> begin performance[i] = Flux.data(modelloss(model(testDS[1]), testDS[2])); end, debug = []);
end

#x = y = linspace(0, 1, 100);
#z = [Flux.data(softmax(model([yi, xi])))[2] for (xi, yi) in Base.product(x, y)];

plotModel(testDS, model, performance);
