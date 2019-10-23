# module test
using GPJ
using DelimitedFiles
using PyPlot

k =  gpflow.kernels.Matern52(1)

data = readdlm(open("../data/regression_1D.csv"), ',')
X = data[:, 1]
Y = data[:, 2]

X = reshape(X, :, 1)
Y = reshape(Y, :, 1)

m =  gpflow.models.GPR(X, Y, k)
# compile!(m)
m.o.likelihood.variance = 0.01 # TODO: Figure out how to remove `.o`
m.o.kern.lengthscales = 0.3

print(m.o.as_pandas_table())

opt = gpflow.train.ScipyOptimizer()
# compile!(opt)
minimize!(opt, m)

print(m.o.as_pandas_table()) # TODO: Output DataFrame

## generate test points for prediction
xx = reshape(range(0,stop=1.1,length=100), 100, 1)  # test points must be of shape (N, D)

## predict mean and variance of latent GP at test points
mean, var = predict_f(m, xx)

## generate 10 samples from posterior
samples = predict_f_samples(m, xx, 10)  # shape (10, 100, 1)

## plot
figure(figsize=(12, 6));
plot(X, Y, "kx", mew=2);
plot(xx, mean, "C0", lw=2);
fill_between(xx[:,1],
                 mean[:,1] - 1.96 * sqrt.(var[:,1]),
                 mean[:,1] + 1.96 * sqrt.(var[:,1]),
                 color="C0", alpha=0.2);
plot(xx, samples[:, :, 1]', "C0", linewidth=.5);
xlim(-0.1, 1.1);
savefig("regression.png");

# end