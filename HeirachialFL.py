import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax import linen as nn
from flax.training import train_state
import optax
from sklearn.metrics import accuracy_score

# Define the neural network model
class SimpleNN(nn.Module):
    @nn.compact
    def __call__(self, x, deterministic):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.2)(x, deterministic=deterministic)
        x = nn.Dense(10)(x)
        return x

# Function to create a training state
def create_train_state(rng, learning_rate, model, input_shape):
    params = model.init(rng, jnp.ones(input_shape), deterministic=True)['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Define the loss and training step functions
def compute_loss(params, apply_fn, x, y, dropout_rng):
    logits = apply_fn({'params': params}, x, rngs={'dropout': dropout_rng})
    one_hot = jax.nn.one_hot(y, num_classes=10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits, one_hot))
    return loss

def train_step(state, batch, dropout_rng):
    def loss_fn(params):
        return compute_loss(params, state.apply_fn, batch[0], batch[1], dropout_rng)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

# Define Client, Edge, and Cloud classes
class Client:
    def __init__(self, id, data):
        self.id = id
        self.data = data
        self.state = None

    def train(self, initial_state, local_epochs, batch_size, learning_rate, dropout_rng):
        if self.state is None:
            self.state = initial_state
        for _ in range(local_epochs):
            for batch_X, batch_y in self.data:
                batch_dropout_rng, dropout_rng = jax.random.split(dropout_rng)
                self.state, loss = train_step(self.state, (batch_X, batch_y), batch_dropout_rng)

    def update_state(self, new_state, dropout_rng):
        self.state = new_state

class Edge:
    def __init__(self, id, clients):
        self.id = id
        self.clients = clients
        self.state = None

    def aggregate(self, states, weights, dropout_rng):
        # Filter out None states and corresponding weights
        valid_states = [state for state in states if state is not None]
        valid_weights = [weight for state, weight in zip(states, weights) if state is not None]
        if valid_states:
            weighted_state = weighted_average_states(valid_states, valid_weights)
            self.state = weighted_state
            
            # Update client states with local fine-tuning
            for client in self.clients:
                client_dropout_rng, dropout_rng = jax.random.split(dropout_rng)
                client.update_state(self.state, client_dropout_rng)
            return self.state
        return None

    def update_state(self, new_state, dropout_rng):
        if new_state is not None:
            self.state = new_state
            for client in self.clients:
                client.update_state(new_state, dropout_rng)

class Cloud:
    def __init__(self, edges):
        self.edges = edges
        self.state = None

    def aggregate(self, states, weights, dropout_rng):
        # Filter out None states and corresponding weights
        valid_states = [state for state in states if state is not None]
        valid_weights = [weight for state, weight in zip(states, weights) if state is not None]
        
        # Weighted average of valid states
        if valid_states:
            weighted_state = weighted_average_states(valid_states, valid_weights)
            self.state = weighted_state
            
            # Update edge states
            for edge in self.edges:
                if edge.state is not None:
                    edge_dropout_rng, dropout_rng = jax.random.split(dropout_rng)
                    edge.update_state(self.state, edge_dropout_rng)
            return self.state
        return None

def weighted_average_states(states, weights):
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    weighted_params = jtu.tree_map(
        lambda *xs: sum(w * x for w, x in zip(normalized_weights, xs)),
        *[state.params for state in states]
    )
    return states[0].replace(params=weighted_params)

# Initialize clients, edges, and cloud
def initialize_clients_and_edges():
    clients = [Client(id=i, data=[(X, y)]) for i in range(num_clients)]
    edges = [Edge(id=i, clients=clients[i*clients_per_edge:(i+1)*clients_per_edge]) for i in range(num_edges)]
    cloud = Cloud(edges=edges)
    return clients, edges, cloud

# Main training loop
rng = jax.random.PRNGKey(0)
learning_rate = 0.001
input_shape = (1, 784)  # Example shape for MNIST
model = SimpleNN()
initial_state = create_train_state(rng, learning_rate, model, input_shape)
dropout_rng = rng

clients, edges, cloud = initialize_clients_and_edges()
for edge in edges:
    edge.state = initial_state

num_epochs = 5
for epoch in range(num_epochs):
    rng, dropout_rng = jax.random.split(rng)
    
    # Local training for each client
    for client in clients:
        client_dropout_rng, dropout_rng = jax.random.split(dropout_rng)
        client.train(initial_state, local_epochs=1, batch_size=32, learning_rate=learning_rate, dropout_rng=client_dropout_rng)

    # Edge aggregation
    edge_states = [edge.state for edge in edges if edge.state is not None]
    edge_weights = [sum(client.data_size for client in edge.clients) for edge in edges if edge.state is not None]
    
    # Cloud aggregation
    global_state = cloud.aggregate(edge_states, edge_weights, dropout_rng)

    # Evaluate on test data
    logits = global_state.apply_fn({'params': global_state.params}, X_test, rngs={'dropout': dropout_rng})
    y_pred = jnp.argmax(logits, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Epoch {epoch + 1}, Accuracy: {accuracy * 100:.2f}%')
