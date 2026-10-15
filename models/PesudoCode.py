        # labels= 1
        # lambda_1 =1
        # lambda_2 = 1
        #
        #
        # '''
        # Pseudo Code of Our Framework
        # Hyper-parameters:
        #     B: batch_size=32, T: sequence_length=8, N: part_number=8,
        #     C: original_channel_number: 2048, C1: reduced_channel_number: 512
        #     lambda_1 = 0.01, lambda_2 = 1.
        # Input:
        #     videos: BT*3*256*128
        # '''
        # ## Step 1: Utilizing backbone network (ResNet-50) to obtain frame-level feature maps
        # x_frame = self.cnn_backbone(videos)  ## x_frame: BT*C*16*8
        # ## Step 2: Yield different partial features.
        # part_feat_1, ar_loss = self.PartGeneration(x_frame) ## part_feat_1: BT*C1*N
        # ## Step 3: Refine each partial feature.
        # part_feat_2 = self.PartRefinement(part_feat_1, x_frame) ## part_feat_2: BT*C1*N
        # ## Step 4: Integrate partial features across frames.
        # part_feat_3 = self.PartEnhencedTransformer(part_feat_2) ## part_feat_3: B*T*C1*N
        # ## Step 5: Constrain the learning partial features.
        # str_loss = self.SpartialTemporalRegularization(part_feat_1, part_feat_2, part_feat_3)
        # ## Step 6: Partial features Supervision.
        # feat = part_feat_3.mean(dim=1).flatten(1) ## feat: B*NC1
        # feat = self.BN(feat) ## feat: B*NC1, utilizing to test.
        # bht_loss = self.BatchHardTripletLoss(feat, labels)
        # classes_score = self.Classifier(feat)
        # ce_loss = self.CrossEntropyLoss(classes_score, labels)
        # total_loss = ce_loss + bht_loss + lambda_1*ar_loss + lambda_2*str_loss