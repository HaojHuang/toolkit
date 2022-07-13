# pts: (N,3), g: (4,4)
pts = np.matmul(grasp_pc, g[:3, :3].T)
pts += np.expand_dims(g[:3, 3], 0)

# same as above (R*P^T)^T = P*R^T
gripper_points_sim = gripper_points_sim.unsqueeze(dim=0).repeat(len(trans),1,1)
gripper_points_sim = torch.einsum('pij,pjk->pik', trans[:,:3,:3],gripper_points_sim.transpose(1,2))
gripper_points_sim = gripper_points_sim.transpose(1,2)
#print(gripper_points_sim.size())
gripper_points_sim = gripper_points_sim + trans[:,:3,-1].unsqueeze(dim=1).repeat(1,num_p,1)

#pre rotation vs after roation: http://www.me.unm.edu/~starr/teaching/me582/postmultiply.pdf
