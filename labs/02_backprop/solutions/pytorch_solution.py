# Create random input and output data
X = torch.randn(dimensions[0], N_batch)
y = torch.randn(dimensions[-1], N_batch)

# Randomly initialize weights & biases
W1 = torch.randn(dimensions[1], dimensions[0], requires_grad=True)
W2 = torch.randn(dimensions[2], dimensions[1], requires_grad=True)
b1 = torch.randn(dimensions[1], 1, requires_grad=True)
b2 = torch.randn(dimensions[2], 1, requires_grad=True)

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
    loss.backward()

    # Update weights & biases
    with torch.no_grad():
        W1 -= eta * W1.grad
        b1 -= eta * b1.grad
        W2 -= eta * W2.grad
        b2 -= eta * b2.grad
        # Manually zero the gradients after updating weights
        W1.grad.zero_()
        W2.grad.zero_()
        b1.grad.zero_()
        b2.grad.zero_()
