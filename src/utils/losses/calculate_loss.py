def calculate_loss(self, gen_img, gen_img_stg1, gen_logits, gen_logits_stg1, 
            real_img, real_logits, real_logits_stg1, compute_Dr1=False):
            
    """모델 출력값으로부터 모든 손실 계산"""
    # Generator losses
    loss_Gmain = torch.nn.functional.softplus(-gen_logits).mean()
    loss_Gmain_stg1 = torch.nn.functional.softplus(-gen_logits_stg1).mean()
    pcp_loss, _ = self.pcp(gen_img, real_img)
    l1_loss = torch.mean(torch.abs(gen_img - real_img))

    # Discriminator fake losses
    loss_Dgen = torch.nn.functional.softplus(gen_logits).mean()
    loss_Dgen_stg1 = torch.nn.functional.softplus(gen_logits_stg1).mean()

    # Discriminator real losses
    loss_Dreal = torch.nn.functional.softplus(-real_logits).mean()
    loss_Dreal_stg1 = torch.nn.functional.softplus(-real_logits_stg1).mean()

    # R1 regularization 계산
    loss_Dr1, loss_Dr1_stg1 = 0.0, 0.0
    if compute_Dr1 and real_img.requires_grad:
        # Main path
        r1_grads = torch.autograd.grad(
            outputs=real_logits.sum(),
            inputs=real_img,
            create_graph=True,
            retain_graph=True
        )[0]
        r1_penalty = r1_grads.square().sum([1,2,3]).mean()
        loss_Dr1 = r1_penalty * (self.r1_gamma / 2)

        # Stage1 path
        r1_grads_stg1 = torch.autograd.grad(
            outputs=real_logits_stg1.sum(),
            inputs=real_img,
            create_graph=True,
            retain_graph=False
        )[0]
        r1_penalty_stg1 = r1_grads_stg1.square().sum([1,2,3]).mean()
        loss_Dr1_stg1 = r1_penalty_stg1 * (self.r1_gamma / 2)

    return {
        # Generator
        'G/total': (loss_Gmain + loss_Gmain_stg1 
                + pcp_loss * self.pcp_ratio),
        'G/main': loss_Gmain,
        'G/stage1': loss_Gmain_stg1,
        'G/pcp': pcp_loss,
        'G/l1': l1_loss,
        
        # Discriminator (Fake)
        'D/fake': loss_Dgen + loss_Dgen_stg1,
        'D/fake_main': loss_Dgen,
        'D/fake_stage1': loss_Dgen_stg1,
        
        # Discriminator (Real)
        'D/real': loss_Dreal + loss_Dreal_stg1,
        'D/real_main': loss_Dreal,
        'D/real_stage1': loss_Dreal_stg1,
        
        # Regularization
        'D/r1': loss_Dr1 + loss_Dr1_stg1,
        'D/r1_main': loss_Dr1,
        'D/r1_stage1': loss_Dr1_stg1
    }