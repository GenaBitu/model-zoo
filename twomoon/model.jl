using Flux;
using Plots;

include("dataset.jl")

function plotModel(ds::Tuple{AbstractMatrix, Flux.OneHotMatrix}, model, performance)
	p1 = plotTwoMoonDS(testDS);
	heatmap!(p1, x, y, z, colorbar = false);
	p2 = plot(performance, label="Loss", ylims = (0:1));
	plot(p1, p2, layout=(1,2));
end

numBatches = 1000;
σ = Flux.relu;
loss = Flux.mse;
modelloss(x, y) = Flux.mse(softmax(x), y);
#reg = Flux.l2(0.000001);
#reg = Flux.regcov(0.00001);
reg = Flux.l2(0);

#Layer(in::Int, out::Int) = Target(Chain(Dense(in, 8, σ), Dense(8, 8, σ), Dense(8, out)), Chain(Dense(out, 8, σ), Dense(8, 8, σ), Dense(8, in, identity)), loss; regulariser = reg);
Layer(in::Int, out::Int) = Target(Chain(Dense(in, out, σ)), Chain(Dense(out, 8, σ), Dense(8, in, identity)), loss; regulariser = reg);

#model = Chain(Layer(2, 4), Layer(4, 8), Layer(8, 8), Layer(8, 4), Target(4, 2, tanh, loss; regulariser = reg));
#model = Chain(Layer(2, 16), Layer(16, 2));
model = Chain(Layer(2, 2));
trainDS = generateTwoMoonDS(10);
testDS = generateTwoMoonDS(100);

performance = Vector{Float32}(numBatches);

for i in 1:numBatches
	targettrain!(model, modelloss, [trainDS], ADAM(params(model)), η = 0.01, cb = () -> begin performance[i] = Flux.data(modelloss(model(testDS[1]), testDS[2])); end);
end

x = 0:0.01:1;
y = 0:0.01:1;
z = [Flux.data(model([yi, xi]))[2] for (xi, yi) in Base.product(x, y)];

plotModel(testDS, model, performance);
