import torch
import matplotlib.pyplot as plt

EPOCHS = 500
BATCH_SIZE = 100000   # büyük tut ki dağılım net görünsün


def sample_t(current_epoch):
    progress = current_epoch / EPOCHS

    mix = min(progress / 0.5, 1.0)

    u = torch.rand(BATCH_SIZE)
    ln = torch.sigmoid(torch.randn(BATCH_SIZE))

    t = (1 - mix) * ln + mix * u

    return t.numpy()


# İncelenecek epochlar
epochs_to_plot = [1, 50, 125, 180, 250, 500]

fig, axes = plt.subplots(1, len(epochs_to_plot), figsize=(20, 4))

for ax, epoch in zip(axes, epochs_to_plot):
    samples = sample_t(epoch)

    ax.hist(samples, bins=100, density=True)
    ax.set_title(f"Epoch {epoch}")
    ax.set_xlim(0, 1)

plt.tight_layout()
plt.tight_layout()
plt.savefig("timestep_sampling.png", dpi=300)
print("Saved figure to timestep_sampling.png")