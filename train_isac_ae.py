"""
ISAC End-to-End Autoencoder Training (PyTorch)

WORKFLOW:
  1. Run AE.m in MATLAB first - generates isac_data.mat
  2. Run this script - trains the model and shows evaluation results

Usage:
  python train_isac_ae.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io
import h5py
import numpy as np
import matplotlib.pyplot as plt

# ========== CONFIGURATION ==========
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

BATCH_SIZE = 512
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
OMEGA_R = 0.09  # ISAC trade-off

# ========== NEURAL NETWORK ARCHITECTURES (Table I) ==========

class PresenceDetector(nn.Module):
    """Detects target presence from radar return"""
    def __init__(self, K):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2*K, 2*K), nn.ReLU(),
            nn.Linear(2*K, 2*K), nn.ReLU(),
            nn.Linear(2*K, K), nn.ReLU(),
            nn.Linear(K, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

class AngleEstimator(nn.Module):
    """Estimates target AoA"""
    def __init__(self, K):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2*K, 2*K), nn.ReLU(),
            nn.Linear(2*K, 2*K), nn.ReLU(),
            nn.Linear(2*K, K), nn.ReLU(),
            nn.Linear(K, 1), nn.Tanh()
        )
    
    def forward(self, x):
        return self.net(x) * (np.pi / 2)

class UncertaintyEstimator(nn.Module):
    """Estimates uncertainty in angle"""
    def __init__(self, K):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2*K, 2*K), nn.ReLU(),
            nn.Linear(2*K, 2*K), nn.ReLU(),
            nn.Linear(2*K, K), nn.ReLU(),
            nn.Linear(K, 1), nn.Softplus()
        )
    
    def forward(self, x):
        return self.net(x) + 0.01

class CommReceiver(nn.Module):
    """Recovers message from received signal with CSI"""
    def __init__(self, M, K):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, K), nn.ReLU(),  # [z_c_real, z_c_imag, kappa_real, kappa_imag]
            nn.Linear(K, 2*K), nn.ReLU(),
            nn.Linear(2*K, 2*K), nn.ReLU(),
            nn.Linear(2*K, M)
        )
    
    def forward(self, x):
        return self.net(x)

class ISACReceiver(nn.Module):
    """Complete ISAC Receiver (Radar + Comm)"""
    def __init__(self, K, M):
        super().__init__()
        self.presence_detector = PresenceDetector(K)
        self.angle_estimator = AngleEstimator(K)
        self.uncertainty_estimator = UncertaintyEstimator(K)
        self.comm_receiver = CommReceiver(M, K)
    
    def forward(self, zr_feat, zc_feat):
        q = self.presence_detector(zr_feat)
        theta_hat = self.angle_estimator(zr_feat)
        sigma_hat = self.uncertainty_estimator(zr_feat)
        m_logits = self.comm_receiver(zc_feat)
        return q, theta_hat, sigma_hat, m_logits

# ========== LOSS FUNCTION ==========
def compute_loss(q, theta_hat, sigma_hat, m_logits, t, theta_true, m_true, omega_r=0.09):
    eps = 1e-7
    
    # Radar detection loss (BCE)
    loss_det = -torch.mean(t * torch.log(q + eps) + (1 - t) * torch.log(1 - q + eps))
    
    # Radar regression loss (NLL)
    nll = torch.log(sigma_hat + eps) + 0.5 * ((theta_true - theta_hat)**2) / (sigma_hat**2 + eps)
    loss_reg = torch.mean(t * nll)
    
    # Comm loss (CE)
    loss_comm = nn.CrossEntropyLoss()(m_logits, m_true)
    
    # Combined ISAC loss
    loss = omega_r * (loss_det + 0.5 * loss_reg) + (1 - omega_r) * loss_comm
    
    return loss, loss_det.item(), loss_reg.item(), loss_comm.item()

# ========== DATA LOADING ==========
def load_data(filepath='isac_data.mat'):
    print(f"Loading {filepath}...")
    
    try:
        # Try HDF5 format (MATLAB v7.3)
        with h5py.File(filepath, 'r') as f:
            train = type('obj', (object,), {})()
            test = type('obj', (object,), {})()
            params = type('obj', (object,), {})()
            
            # Helper function to convert structured arrays with real/imag fields
            def convert_structured_complex(val):
                if isinstance(val, np.ndarray) and val.dtype.names == ('real', 'imag'):
                    # Reconstruct complex array
                    return val['real'] + 1j * val['imag']
                return val
            
            # Load train data
            for key in f['train'].keys():
                val = f['train'][key][()]
                # Transpose if needed
                if isinstance(val, np.ndarray) and val.ndim > 1:
                    val = val.T
                # Convert structured complex arrays
                val = convert_structured_complex(val)
                # Convert single element arrays
                if isinstance(val, np.ndarray) and val.size == 1 and key not in ['Y_radar', 'Y_comm', 'kappa', 'alpha', 'beta']:
                    try:
                        val = val.item()
                    except:
                        pass
                setattr(train, key, val)
            
            # Load test data
            for key in f['test'].keys():
                val = f['test'][key][()]
                # Transpose if needed
                if isinstance(val, np.ndarray) and val.ndim > 1:
                    val = val.T
                # Convert structured complex arrays
                val = convert_structured_complex(val)
                # Convert single element arrays
                if isinstance(val, np.ndarray) and val.size == 1 and key not in ['Y_radar', 'Y_comm', 'kappa', 'alpha', 'beta']:
                    try:
                        val = val.item()
                    except:
                        pass
                setattr(test, key, val)
            
            # Load params
            for key in f['params'].keys():
                val = f['params'][key][()]
                if isinstance(val, np.ndarray) and val.size == 1:
                    val = val.item()
                setattr(params, key, val)
    except (OSError, KeyError) as e:
        print(f"Error with HDF5 loading: {e}")
        # Fallback to scipy for older MATLAB format
        mat = scipy.io.loadmat(filepath, squeeze_me=True, struct_as_record=False)
        train = mat['train']
        test = mat['test']
        params = mat['params']
    
    return train, test, params

def prepare_tensors(data, device):
    tensors = {}
    
    # Message (0-indexed for PyTorch)
    m_idx = np.array(data.m_idx).squeeze()
    tensors['m_idx'] = torch.tensor(m_idx - 1, dtype=torch.long, device=device)
    
    # Target presence
    target_present = np.array(data.target_present).squeeze()
    tensors['t'] = torch.tensor(target_present, dtype=torch.float32, device=device).unsqueeze(1)
    
    # Angles
    theta_true = np.array(data.theta_true).squeeze()
    tensors['theta'] = torch.tensor(theta_true, dtype=torch.float32, device=device).unsqueeze(1)
    
    # Radar features
    Y_radar = np.array(data.Y_radar, dtype=np.complex128)
    Y_radar_tensor = torch.tensor(Y_radar, dtype=torch.complex64, device=device)
    tensors['zr_feat'] = torch.cat([Y_radar_tensor.real, Y_radar_tensor.imag], dim=1)
    
    # Comm features with CSI
    Y_comm = np.array(data.Y_comm, dtype=np.complex128).squeeze()
    kappa = np.array(data.kappa, dtype=np.complex128).squeeze()
    
    Y_comm_tensor = torch.tensor(Y_comm, dtype=torch.complex64, device=device)
    kappa_tensor = torch.tensor(kappa, dtype=torch.complex64, device=device)
    
    if Y_comm_tensor.dim() == 1:
        Y_comm_tensor = Y_comm_tensor.unsqueeze(1)
    if kappa_tensor.dim() == 1:
        kappa_tensor = kappa_tensor.unsqueeze(1)
    
    tensors['zc_feat'] = torch.cat([Y_comm_tensor.real, Y_comm_tensor.imag, kappa_tensor.real, kappa_tensor.imag], dim=1)
    
    return tensors

# ========== TRAINING ==========
def train_model(model, train_tensors, num_epochs, batch_size, lr, omega_r):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    N = train_tensors['m_idx'].shape[0]
    loss_history = []
    
    print(f"\nTraining: {num_epochs} epochs, batch={batch_size}, lr={lr}, omega_r={omega_r}")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        model.train()
        perm = torch.randperm(N)
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, N, batch_size):
            idx = perm[i:min(i+batch_size, N)]
            
            q, theta_hat, sigma_hat, m_logits = model(
                train_tensors['zr_feat'][idx],
                train_tensors['zc_feat'][idx]
            )
            
            loss, _, _, _ = compute_loss(
                q, theta_hat, sigma_hat, m_logits,
                train_tensors['t'][idx],
                train_tensors['theta'][idx],
                train_tensors['m_idx'][idx],
                omega_r
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    
    return loss_history

# ========== EVALUATION ==========
def evaluate(model, test_tensors, P_fa_target=0.01):
    model.eval()
    
    with torch.no_grad():
        q, theta_hat, sigma_hat, m_logits = model(
            test_tensors['zr_feat'],
            test_tensors['zc_feat']
        )
        
        t = test_tensors['t'].squeeze()
        q = q.squeeze()
        
        # Detection threshold
        q_h0 = q[t == 0]
        threshold = torch.quantile(q_h0, 1 - P_fa_target) if len(q_h0) > 0 else 0.5
        t_hat = (q > threshold).float()
        
        # Detection probability
        Pd = (t_hat[t == 1] == 1).float().mean().item() if (t == 1).sum() > 0 else 0
        Pfa = (t_hat[t == 0] == 1).float().mean().item() if (t == 0).sum() > 0 else 0
        
        # RMSE
        detected = (t == 1) & (t_hat == 1)
        if detected.sum() > 0:
            RMSE = torch.sqrt(torch.mean((theta_hat.squeeze()[detected] - test_tensors['theta'].squeeze()[detected])**2)).item()
            sigma_mean = sigma_hat.squeeze()[detected].mean().item()
        else:
            RMSE = float('nan')
            sigma_mean = float('nan')
        
        # SER
        m_pred = torch.argmax(m_logits, dim=1)
        SER = (m_pred != test_tensors['m_idx']).float().mean().item()
        
    return SER, Pd, Pfa, RMSE, sigma_mean

# ========== MAIN ==========
if __name__ == "__main__":
    # Load data
    try:
        train_data, test_data, params = load_data('isac_data.mat')
    except FileNotFoundError:
        print("=" * 60)
        print("ERROR: isac_data.mat not found!")
        print("Please run AE.m in MATLAB first to generate the data.")
        print("=" * 60)
        exit(1)
    
    K = int(params.K)
    M = int(params.M)
    
    print(f"Loaded: K={K}, M={M}")
    print(f"Training samples: {len(train_data.m_idx)}")
    print(f"Test samples: {len(test_data.m_idx)}")
    
    # Prepare tensors
    train_tensors = prepare_tensors(train_data, DEVICE)
    test_tensors = prepare_tensors(test_data, DEVICE)
    
    # Create model
    model = ISACReceiver(K, M).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    loss_history = train_model(model, train_tensors, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, OMEGA_R)
    
    # ========== EVALUATION ==========
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    SER, Pd, Pfa, RMSE, sigma_mean = evaluate(model, test_tensors)
    
    print(f"\nSymbol Error Rate (SER):     {SER:.4f}")
    print(f"Detection Probability (Pd):   {Pd:.4f}")
    print(f"False Alarm Rate (Pfa):       {Pfa:.4f}")
    print(f"Angle RMSE:                   {RMSE:.4f} rad ({np.degrees(RMSE):.2f} deg)")
    print(f"Mean Uncertainty:             {sigma_mean:.4f} rad")
    
    # ========== PLOTS ==========
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training loss
    axes[0, 0].plot(loss_history, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True)
    
    # Results bar chart
    metrics = ['SER', 'Pd', 'Pfa']
    values = [SER, Pd, Pfa]
    colors = ['#e74c3c', '#2ecc71', '#f39c12']
    axes[0, 1].bar(metrics, values, color=colors)
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].set_title('Performance Metrics')
    axes[0, 1].grid(True, axis='y')
    
    # RMSE vs Uncertainty
    with torch.no_grad():
        q, theta_hat, sigma_hat, _ = model(test_tensors['zr_feat'], test_tensors['zc_feat'])
        t = test_tensors['t'].squeeze()
        detected = (t == 1)
        
        if detected.sum() > 0:
            th_err = torch.abs(theta_hat.squeeze()[detected] - test_tensors['theta'].squeeze()[detected])
            sig = sigma_hat.squeeze()[detected]
            
            axes[1, 0].scatter(sig.cpu().numpy(), th_err.cpu().numpy(), alpha=0.1, s=5)
            axes[1, 0].plot([0, 0.5], [0, 0.5], 'r--', linewidth=2, label='RMSE = σ')
            axes[1, 0].set_xlabel('Estimated Uncertainty σ [rad]')
            axes[1, 0].set_ylabel('Actual Error |θ - θ̂| [rad]')
            axes[1, 0].set_title('Uncertainty Calibration')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
    
    # Detection histogram
    with torch.no_grad():
        q_h1 = q.squeeze()[t == 1].cpu().numpy()
        q_h0 = q.squeeze()[t == 0].cpu().numpy()
        
        axes[1, 1].hist(q_h0, bins=50, alpha=0.6, label='H0 (No Target)', color='blue')
        axes[1, 1].hist(q_h1, bins=50, alpha=0.6, label='H1 (Target)', color='red')
        axes[1, 1].axvline(x=0.5, color='black', linestyle='--', label='Threshold')
        axes[1, 1].set_xlabel('Detection Score q')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Detection Score Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('ISAC_Results.png', dpi=150)
    print("\nPlots saved to ISAC_Results.png")
    
    # Save model
    torch.save(model.state_dict(), 'isac_model.pth')
    print("Model saved to isac_model.pth")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)