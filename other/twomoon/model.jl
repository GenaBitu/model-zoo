using Flux;
using PGFPlots;
using Colors;

include("dataset.jl")

function plotModel(ds::Tuple{AbstractMatrix, Flux.OneHotMatrix}, model, performance)
	plots = [plotTwoMoonDS(testDS)];
	push!(plots[1], Plots.Image((x,y)->Flux.data(softmax(model([y, x])))[2], (0,1), (0, 1), colormap = ColorMaps.RGBArrayMap(colormap("RdBu"), invert = true, interpolation_levels= 500), zmin = 0, zmax = 1));
	#push!(plots, plot(performance, label = "Loss", ylims = (0.001,1), yscale = :log10, title = "Final loss"));
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
						jacobian .= value .* jacobian;
					end
				end
			end
		end
		i += 1
	end
	if size(jacobian) != (0,)
		singularvalues = map(x->svdvals(x'), jacobian);
		ratios = map(x->x[1] / x[end], singularvalues);
		#push!(plots, plot(acosd.(1 ./ ratios), label = "", title = "Maximum angle by theorem", ylims = (0,180),))
	end
	plot(plots[1])
end

numBatches = 2000;
σ = Flux.swish;
noiseDeviation = 0.2;
loss = Flux.mse;
modelloss(x, y) = Flux.mse(softmax(x), y);
#reg = Flux.l2(0.000001);
#reg = Flux.regcov(0.00001);
#reg = Flux.l2(0);

Layer(in::Int, out::Int) = Target(Chain(Dense(in, out, σ)), Chain(Dense(out, 8, σ), Dense(8, in, identity)), loss; σ = noiseDeviation);

#model = Chain(Layer(2, 4), Layer(4, 8), Layer(8, 8), Layer(8, 4), Target(4, 2, tanh, loss; regulariser = reg));
model = Chain(Layer(2, 16), Layer(16, 2));
trainDS = generateTwoMoonDS(1000);
testDS = generateTwoMoonDS(100);

performance = Vector{Float32}(numBatches);

for i in 1:numBatches
	targettrain!(model, modelloss, [trainDS], ADAM(params(model)), η = 0.5, cb = () -> begin performance[i] = Flux.data(modelloss(model(testDS[1]), testDS[2])); end, debug = ["Classifier loss", "Auto-encoder loss", "angle"]);
end

#x = y = linspace(0, 1, 100);
#z = [Flux.data(model([yi, xi]))[2] for (xi, yi) in Base.product(x, y)];

plotModel(testDS, model, performance);
