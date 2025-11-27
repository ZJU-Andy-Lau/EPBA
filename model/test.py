import torch
import unittest
import math
from encoder import Rope, SelfAttentionBlock, CrossAttentionBlock, Adapter, Encoder

class TestEncoderModules(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.B = 2
        self.H = 32
        self.W = 32
        self.embed_dim = 64 # 小维度方便测试
        self.num_heads = 4
        self.head_dim = 16

    def test_1_rope_correctness(self):
        """测试 RoPE: 2D 编码逻辑与数值正交性"""
        print("\n[Test 1] Testing 2D RoPE...")
        rope = Rope(d_model=self.head_dim).to(self.device)
        
        dummy_x = torch.randn(self.B, self.embed_dim, self.H, self.W).to(self.device)
        cos, sin = rope(dummy_x)
        
        # 1. 形状检查 [1, 1, L, D]
        self.assertEqual(cos.shape, (1, 1, self.H * self.W, self.head_dim))
        
        # 2. 三角恒等式: cos^2 + sin^2 = 1
        identity = cos**2 + sin**2
        self.assertTrue(torch.allclose(identity, torch.ones_like(identity), atol=1e-5))
        print("  > RoPE trigonometric check passed.")

    def test_2_attention_blocks(self):
        """测试 Self 和 Cross Attention 模块"""
        print("\n[Test 2] Testing Attention Blocks...")
        
        self_attn = SelfAttentionBlock(self.embed_dim, self.num_heads).to(self.device)
        cross_attn = CrossAttentionBlock(self.embed_dim, self.num_heads).to(self.device)
        
        # 模拟数据
        seq_len = self.H * self.W
        x = torch.randn(self.B, seq_len, self.embed_dim).to(self.device)
        ctx = torch.randn(self.B, seq_len, self.embed_dim).to(self.device)
        
        # 模拟 RoPE
        rope = Rope(self.head_dim).to(self.device)
        cos, sin = rope(torch.randn(self.B, self.embed_dim, self.H, self.W).to(self.device))
        
        # Self Attn Forward
        out_self = self_attn(x, cos, sin)
        self.assertEqual(out_self.shape, x.shape)
        
        # Cross Attn Forward
        out_cross = cross_attn(x, ctx, cos, sin)
        self.assertEqual(out_cross.shape, x.shape)
        
        print("  > Attention blocks forward passed.")

    def test_3_adapter_interaction(self):
        """测试 Adapter 的双流交互逻辑"""
        print("\n[Test 3] Testing Adapter Interaction...")
        
        # 模拟 DINO 输出维度 (1024)
        input_dim = 1024
        adapter = Adapter(input_dim=input_dim, embed_dim=64, num_layers=2).to(self.device)
        
        # 模拟两个不同的特征图
        feat0 = torch.randn(self.B, input_dim, 16, 16).to(self.device)
        feat1 = torch.randn(self.B, input_dim, 16, 16).to(self.device)
        
        # Forward
        m0, c0, conf0, m1, c1, conf1 = adapter(feat0, feat1)
        
        # 1. 形状检查
        self.assertEqual(m0.shape, (self.B, 64, 16, 16))
        self.assertEqual(c0.shape, (self.B, 128, 16, 16)) # ctx_dim默认128
        self.assertEqual(conf0.shape, (self.B, 1, 16, 16))
        
        # 2. L2 归一化检查 (Match Feature 必须归一化)
        norm0 = torch.norm(m0, p=2, dim=1)
        self.assertTrue(torch.allclose(norm0, torch.ones_like(norm0), atol=1e-5))
        print("  > L2 Normalization verified.")
        
        # 3. 交互性验证 (Interaction Verification)
        # 只要证明 feat0 的输出受到了 feat1 的影响即可
        # 方法：保持 feat0 不变，改变 feat1，看 m0 是否变化
        
        feat1_changed = torch.randn_like(feat1)
        m0_new, _, _, _, _, _ = adapter(feat0, feat1_changed)
        
        diff = (m0 - m0_new).abs().max().item()
        print(f"  Interaction Difference: {diff}")
        self.assertTrue(diff > 1e-4, "Cross Attention failed! Output m0 is independent of input feat1.")
        print("  > Cross-stream interaction verified.")

    def test_4_encoder_integration(self):
        """
        测试 Encoder 整体流程 (需 Mock DINO 以免下载权重)
        """
        print("\n[Test 4] Testing Encoder Pipeline (Mocked)...")
        
        # 继承并覆盖 _extract_dino_features 方法，避免加载真实模型
        class MockEncoder(Encoder):
            def __init__(self):
                super(Encoder, self).__init__() # Skip Encoder init to avoid loading hub
                self.embed_dim = 64
                self.ctx_dim = 32
                # 初始化 Adapter
                self.adapter = Adapter(input_dim=256, embed_dim=64, ctx_dim=32)
            
            def _extract_dino_features(self, x):
                # 模拟 DINO 输出: [B, 256, H/16, W/16]
                B, _, H, W = x.shape
                return torch.randn(B, 256, H//16, W//16).to(x.device)

        model = MockEncoder().to(self.device)
        
        img0 = torch.randn(self.B, 3, 256, 256).to(self.device)
        img1 = torch.randn(self.B, 3, 256, 256).to(self.device)
        
        (m0, c0, conf0), (m1, c1, conf1) = model(img0, img1)
        
        print(f"  Output Match Shape: {m0.shape}")
        self.assertEqual(m0.shape, (self.B, 64, 16, 16)) # 256/16 = 16
        print("  > Encoder pipeline passed.")

if __name__ == '__main__':
    unittest.main()