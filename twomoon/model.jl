using Flux;
using Plots;

include("dataset.jl")

function plotModel(ds::Tuple{AbstractMatrix, Flux.OneHotMatrix}, model, performance)
	p1 = plotTwoMoonDS(testDS);
	heatmap!(p1, x, y, z, colorbar = false);
	p2 = plot(performance, label="Loss", ylims = (0:1));
	plot(p1, p2, layout=(1,2));
end

numBatches = 2;
loss = Flux.logitcrossentropy;

Layer(in::Int, out::Int) = TargetDense(in, out, Flux.swish, loss);

model = Chain(Layer(2, 4), Layer(4, 8), Layer(8, 8), Layer(8, 4), TargetDense(4, 2, tanh, loss), TargetSoftmax(2, loss));
trainDS = generateTwoMoonDS(1000);
testDS = generateTwoMoonDS(100);

modelloss(x, y) = loss(model(x), y);

performance = Vector{Float32}(numBatches);

for i in 1:numBatches
	targettrain!(model, [trainDS], ADAM(params(model)), cb = () -> begin performance[i] = Flux.data(modelloss(testDS...)); end);
end

x = 0:0.01:1;
y = 0:0.01:1;
z = [Flux.data(model([yi, xi]))[2] for (xi, yi) in Base.product(x, y)];

plotModel(testDS, model, performance);
