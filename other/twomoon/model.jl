using Flux;
using Plots;

include("dataset.jl")

function plotModel(ds::Tuple{AbstractMatrix, Flux.OneHotMatrix}, model, performance)
	plots = [plotTwoMoonDS(testDS)];
	x = y = linspace(0, 1, 100);
	z = [Flux.data(softmax(model([yi, xi])))[2] for (xi, yi) in Base.product(x, y)];
	heatmap!(plots[1], x, y, z, colorbar = false, title = "Heatmap", clims = (0, 1));
	push!(plots, plot(performance, label = "Loss", ylims = (0.001, 1), yscale = :log10, title = "Final loss"));
	return plots;
end

numBatches = 2000;
σ = Flux.relu;
modelloss(x, y) = Flux.mse(softmax(x), y);

Layer(in::Int, out::Int) = Dense(in, out, σ);
model = Chain(Layer(2, 16), Layer(16, 2));

trainDS = generateTwoMoonDS(1000);
testDS = generateTwoMoonDS(1000);

performance = Vector{Float32}(numBatches);

for i in 1:numBatches
	Flux.train!((x, y)->modelloss(model(x), y), [trainDS], ADAM(params(model)), cb = () -> begin performance[i] = Flux.data(modelloss(model(testDS[1]), testDS[2])); end);
end

plotModel(testDS, model, performance);
