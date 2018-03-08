using Flux;

include("dataset.jl")

Layer(in::Int, out::Int) = Flux.Dense(in, out, tanh);

model = Chain(Layer(2, 4), Layer(4, 8), Layer(8, 8), Layer(8, 4), Dense(4, 2), softmax);
trainDS = generateTwoMoonDS(1000);
testDS = generateTwoMoonDS(100);

loss(x, y) = Flux.logitcrossentropy(model(x), Flux.onehotbatch(y, 1:2));

for _ in 1:10000
	Flux.train!(loss, [trainDS], ADAM(params(model)));
end

x = 0:0.01:1;
y = 0:0.01:1;
z = [model([yi, xi]).data[2] for (xi, yi) in Base.product(x, y)];

plotTwoMoonDS(testDS);
heatmap!(x, y, z);
