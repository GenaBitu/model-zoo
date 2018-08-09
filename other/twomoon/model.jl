using Flux;
using Plots;

include("dataset.jl")

function plotModel(ds::Tuple{AbstractMatrix, AbstractVector}, model, performance)
	p1 = plotTwoMoonDS(testDS);
	heatmap!(p1, x, y, z, colorbar = false);
	p2 = plot(performance, label="Loss", ylims = (0:1));
	plot(p1, p2, layout=(1,2));
end

numBatches = 10000;

Layer(in::Int, out::Int) = Flux.Dense(in, out, Flux.swish);

model = Chain(Layer(2, 4), Layer(4, 8), Layer(8, 8), Layer(8, 4), Dense(4, 2, tanh), softmax);
trainDS = generateTwoMoonDS(1000);
testDS = generateTwoMoonDS(100);

loss(x, y) = Flux.logitcrossentropy(model(x), Flux.onehotbatch(y, 1:2));

performance = Vector{Float32}(numBatches);

for i in 1:numBatches
	Flux.train!(loss, [trainDS], ADAM(params(model)), cb = () -> begin performance[i] = Flux.data(loss(testDS...)); end);
end

x = 0:0.01:1;
y = 0:0.01:1;
z = [model([yi, xi]).data[2] for (xi, yi) in Base.product(x, y)];

plotModel(testDS, model, performance);
