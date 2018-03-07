using Flux;

include("dataset.jl")

Layer(in::Int, out::Int) = Flux.Dense(in, out, Flux.relu);

model = Chain(Layer(2, 8), Layer(8, 8), Layer(8, 8), Dense(8, 2), softmax);
trainDS = generateTwoMoonDS(1000);
testDS = generateTwoMoonDS(100);

loss(x, y) = Flux.crossentropy(model(x), [y, 1-y]);

for _ in 1:100
	Flux.train!(loss, trainDS, ADAM(params(model)));
end

x = 0:0.01:1;
y = 0:0.01:1;
z = [model([xi, yi])[1].data[1] for (yi, xi) in Base.product(x, y)];

plotTwoMoonDS(testDS);
heatmap!(x, y, z);
