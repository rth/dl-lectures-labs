N_batch, dimensions = 64, [784, 100, 10]

# Create random input and output data
X = torch.randn(dimensions[0], N_batch)
y = torch.randn(dimensions[-1], N_batch)

# Randomly initialize weights & biases
W1 = torch.randn(dimensions[1], dimensions[0])
W2 = torch.randn(dimensions[2], dimensions[1])
b1 = torch.randn(dimensions[1], 1)
b2 = torch.randn(dimensions[2], 1)

eta, MAXITER, SKIP = 5e-6, 2500, 100
for epoch in range(MAXITER):
    # Forward propagation: compute predicted y
    Z1 = torch.mm(W1, X) + b1
    A1 = torch.relu(Z1)        # Native PyTorch ReLU function
    Z2 = torch.mm(W2, A1) + b2
    A2 = Z2

    # Compute and print loss
    loss = 0.5 * (A2 - y).pow(2).sum()
    if (divmod(epoch, SKIP)[1]==0):
        print(epoch, loss.item())

    # Backpropagation to compute gradients of loss with respect to W1, W2, b1, and b2
    delta2 = (A2 - y)               # derivative of identity map == multiplying by ones
    grad_W2 = torch.mm(delta2, A1.T)
    grad_b2 = torch.mm(delta2, torch.ones(N_batch, 1))
    delta1 = torch.mm(W2.T, delta2) * (Z1>0) # derivative of ReLU is a step function
    grad_W1 = torch.mm(delta1, X.T)
    grad_b1 = torch.mm(delta1, torch.ones(N_batch, 1))

    # Update weights & biases
    W1 = W1 - eta * grad_W1
    b1 = b1 - eta * grad_b1
    W2 = W2 - eta * grad_W2
    b2 = b2 - eta * grad_b2
