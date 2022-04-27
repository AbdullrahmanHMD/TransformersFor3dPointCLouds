
def train(model, optimizer, train_loader, criterion, epochs):
    
    total_loss = []

    for epoch in epochs:
        epoch_loss = 0
        for x, y, _ in train_loader:
            optimizer.zero_grad()

            yhat = model(x)
            loss = criterion(yhat, y)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        total_loss.append(epoch_loss)

    return total_loss

