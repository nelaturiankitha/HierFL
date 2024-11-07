import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import jax.tree_util as jtu 

# Define the Flax 

class MyModel(nn.Module):
    hidden_dim: int
    num_classes: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_classes)(x)
        return x

# Client class
class Client:
    def __init__(self, id, X, y, model):
        self.id = id
        self.X = X
        self.y = y
        self.model = model
        self.upload_cost = 0

    def train(self, state, num_epochs, batch_size):
        for _ in range(num_epochs):
            for i in range(0, len(self.X), batch_size):
                batch_X = self.X[i:i+batch_size]
                batch_y = self.y[i:i+batch_size]
                state, _ = train_step(state, (batch_X, batch_y))
        self.upload_cost += calculate_params_size(state.params)
        return state

# Edge class
class Edge:
    def __init__(self, id, clients):
        self.id = id
        self.clients = clients
        self.enabled = True
        self.download_cost = 0
        self.upload_cost = 0

    def aggregate(self, states):
        if self.enabled:
            self.download_cost += sum(calculate_params_size(state.params) for state in states)
            aggregated_state = average_states(states)
            self.upload_cost += calculate_params_size(aggregated_state.params)
            return aggregated_state
        return None

# Fog class
class Fog:
    def __init__(self, id, edges):
        self.id = id
        self.edges = edges
        self.enabled = True
        self.download_cost = 0
        self.upload_cost = 0

    def aggregate(self, states):
        if self.enabled:
            self.download_cost += sum(calculate_params_size(state.params) for state in states)
            aggregated_state = average_states(states)
            self.upload_cost += calculate_params_size(aggregated_state.params)
            return aggregated_state
        return None

# Cloud class
class Cloud:
    def __init__(self, fogs):
        self.fogs = fogs
        self.download_cost = 0

    def aggregate(self, states):
        self.download_cost += sum(calculate_params_size(state.params) for state in states)
        return average_states(states)

# Helper functions
def create_train_state(rng, learning_rate, model, input_shape):
    params = model.init(rng, jnp.ones(input_shape))
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params['params'], tx=tx)

def compute_loss(params, apply_fn, x, y):
    logits = apply_fn({'params': params}, x)
    one_hot = jax.nn.one_hot(y, num_classes)
    loss = -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=1))
    return loss

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        return compute_loss(params, state.apply_fn, *batch)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def average_states(states):
    if not states:
        return None
    averaged_params = jtu.tree_map(lambda *x: jnp.mean(jnp.stack(x), axis=0), *[state.params for state in states])
    return states[0].replace(params=averaged_params)

def calculate_params_size(params):
    return sum(x.size for x in jax.tree.leaves(params))

# Load and preprocess data
df = pd.read_csv('IOT_TEST.csv')
labelencoder = LabelEncoder()
df['type'] = labelencoder.fit_transform(df['type'])
X = df.drop(['ts', 'label', 'type'], axis=1).values
y = df['type'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337, stratify=y)
X_train, y_train = jnp.array(X_train, dtype=jnp.float32), jnp.array(y_train, dtype=jnp.int32)
X_test, y_test = jnp.array(X_test, dtype=jnp.float32), jnp.array(y_test, dtype=jnp.int32)

# Hyperparameters
num_classes = len(np.unique(y))
hidden_dim = 64
learning_rate = 0.0001
num_epochs = 5
batch_size = 32
num_clients = 10
num_edges = 1
num_fogs = 2

# Initialize model and clients
rng = jax.random.PRNGKey(0)
model = MyModel(hidden_dim=hidden_dim, num_classes=num_classes)
input_shape = (1, X_train.shape[1])
clients = [Client(i, X_train[i::num_clients], y_train[i::num_clients], model) for i in range(num_clients)]

# Create hierarchical structure
edges = [Edge(i, clients[i*3:(i+1)*3]) for i in range(num_edges)]
fogs = [Fog(i, edges[i*2:(i+1)*2]) for i in range(num_fogs)]
cloud = Cloud(fogs)

def set_layer_status(layer, status):
    for item in layer:
        item.enabled = status

set_layer_status(fogs, True)
set_layer_status(edges, True)
edges[0].enabled = True
# Training loop
for epoch in range(num_epochs):
    # Reset costs for this epoch
    for client in clients:
        client.upload_cost = 0
    for edge in edges:
        edge.download_cost = 0
        edge.upload_cost = 0
    for fog in fogs:
        fog.download_cost = 0
        fog.upload_cost = 0
    cloud.download_cost = 0

    # Client training
    client_states = []
    for client in clients:
        state = create_train_state(rng, learning_rate, model, input_shape)
        client_states.append(client.train(state, 1, batch_size))

    # Edge aggregation
    edge_states = []
    for edge in edges:
        if edge.enabled:
            edge_state = edge.aggregate([client_states[client.id] for client in edge.clients])
            if edge_state is not None:
                edge_states.append(edge_state)

    # Fog aggregation
    fog_states = []
    for fog in fogs:
        if fog.enabled:
            enabled_edge_states = [edge_states[edge.id] for edge in fog.edges if edge.enabled and edge.id < len(edge_states)]
            if enabled_edge_states:
                fog_state = fog.aggregate(enabled_edge_states)
                if fog_state is not None:
                    fog_states.append(fog_state)

    # Cloud aggregation
    if fog_states:
        global_state = cloud.aggregate(fog_states)
    elif edge_states:
        global_state = cloud.aggregate(edge_states)
    else:
        global_state = cloud.aggregate(client_states)

    # Calculate and print communication cost
    client_upload = sum(client.upload_cost for client in clients)
    edge_download = sum(edge.download_cost for edge in edges if edge.enabled)
    edge_upload = sum(edge.upload_cost for edge in edges if edge.enabled)
    fog_download = sum(fog.download_cost for fog in fogs if fog.enabled)
    fog_upload = sum(fog.upload_cost for fog in fogs if fog.enabled)
    cloud_download = cloud.download_cost

    total_cost = client_upload + edge_download + edge_upload + fog_download + fog_upload + cloud_download

    print(f"Epoch {epoch+1} Communication Cost:")
    print(f"  Client Upload: {client_upload}")
    print(f"  Edge Download: {edge_download}")
    print(f"  Edge Upload: {edge_upload}")
    print(f"  Fog Download: {fog_download}")
    print(f"  Fog Upload: {fog_upload}")
    print(f"  Cloud Download: {cloud_download}")
    print(f"  Total Cost: {total_cost}")

    # Evaluate the model
    logits = global_state.apply_fn({'params': global_state.params}, X_test)
    y_pred = jnp.argmax(logits, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=1)
    
    print(f"Epoch {epoch+1}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

# Example of disabling layers
set_layer_status(fogs, True)

# You can run the training loop again here with the new configuration if desired