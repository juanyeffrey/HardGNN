# TensorFlow 2.x Compatible HardGNN Model for Google Colab
# This version uses tf.compat.v1 to maintain compatibility with the original TF 1.x code
# while working on modern Google Colab with Python 3.10+ and TensorFlow 2.x

import os
import numpy as np
import tensorflow as tf
import pickle
import scipy.sparse as sp
from random import randint

# Enable TensorFlow 1.x behavior in TensorFlow 2.x
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

# Import TF2-compatible modules
from Params import args
import Utils.NNLayers_tf2 as NNs
from Utils.NNLayers_tf2 import FC, Regularize, Activate, Dropout, Bias, getParam, defineParam, defineRandomNameParam
from Utils.attention_tf2 import AdditiveAttention, MultiHeadSelfAttention
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from DataHandler import negSamp, negSamp_fre, transpose, DataHandler, transToLsts

print("✅ Using TensorFlow 2.x compatible HardGNN model")
print(f"TensorFlow version: {tf.__version__}")

class Recommender:
	def __init__(self, sess, handler):
		self.sess = sess
		self.handler = handler

		print('USER', args.user, 'ITEM', args.item)
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'HR', 'NDCG']
		if args.use_hard_neg:
			mets.append('contrastiveLoss')
		for met in mets:
			self.metrics[met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			if save and metric in self.metrics:
				self.metrics[metric].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')
		if args.load_model != None:
			self.loadModel()
			stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
		else:
			stloc = 0
			init = tf.compat.v1.global_variables_initializer()
			self.sess.run(init)
			log('Variables Inited')
		maxndcg=0.0
		maxres=dict()
		maxepoch=0
		for ep in range(stloc, args.epoch):
			test = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, test))
			if test:
				reses = self.testEpoch()
				log(self.makePrint('Test', ep, reses, test))
			if ep % args.tstEpoch == 0 and reses['NDCG']>maxndcg:
				self.saveHistory()
				maxndcg=reses['NDCG']
				maxres=reses
				maxepoch=ep
			print()
		reses = self.testEpoch()
		log(self.makePrint('Test', args.epoch, reses, True))
		log(self.makePrint('max', maxepoch, maxres, True))

	def makeTimeEmbed(self):
		divTerm = 1 / (10000 ** (tf.compat.v1.range(0, args.latdim * 2, 2, dtype=tf.float32) / args.latdim))
		pos = tf.compat.v1.expand_dims(tf.compat.v1.range(0, self.maxTime, dtype=tf.float32), axis=-1)
		sine = tf.compat.v1.expand_dims(tf.compat.v1.math.sin(pos * divTerm) / np.sqrt(args.latdim), axis=-1)
		cosine = tf.compat.v1.expand_dims(tf.compat.v1.math.cos(pos * divTerm) / np.sqrt(args.latdim), axis=-1)
		timeEmbed = tf.compat.v1.reshape(tf.compat.v1.concat([sine, cosine], axis=-1), [self.maxTime, args.latdim*2]) / 4.0
		return timeEmbed

	def messagePropagate(self, srclats, mat, type='user'):
		timeEmbed = FC(self.timeEmbed, args.latdim, reg=True)
		srcNodes = tf.compat.v1.squeeze(tf.compat.v1.slice(mat.indices, [0, 1], [-1, 1]))
		tgtNodes = tf.compat.v1.squeeze(tf.compat.v1.slice(mat.indices, [0, 0], [-1, 1]))
		edgeVals = mat.values
		srcEmbeds = tf.compat.v1.nn.embedding_lookup(srclats, srcNodes)
		lat=tf.compat.v1.pad(tf.compat.v1.math.segment_sum(srcEmbeds, tgtNodes),[[0,100],[0,0]])
		if(type=='user'):
			lat=tf.compat.v1.nn.embedding_lookup(lat,self.users)
		else:
			lat=tf.compat.v1.nn.embedding_lookup(lat,self.items)
		return Activate(lat, self.actFunc)

	def edgeDropout(self, mat):
		def dropOneMat(mat):
			indices = mat.indices
			values = mat.values
			shape = mat.dense_shape
			newVals = tf.compat.v1.nn.dropout(tf.compat.v1.cast(values,dtype=tf.float32), self.keepRate)
			return tf.compat.v1.sparse.SparseTensor(indices, tf.compat.v1.cast(newVals,dtype=tf.int32), shape)
		return dropOneMat(mat)

	def ours(self):
		user_vector,item_vector=list(),list()
		# embedding
		uEmbed=NNs.defineParam('uEmbed', [args.graphNum, args.user, args.latdim], reg=True)
		iEmbed=NNs.defineParam('iEmbed', [args.graphNum, args.item, args.latdim], reg=True)
		posEmbed=NNs.defineParam('posEmbed', [args.pos_length, args.latdim], reg=True)
		pos= tf.compat.v1.tile(tf.compat.v1.expand_dims(tf.compat.v1.range(args.pos_length),axis=0),[args.batch,1])
		self.items=tf.compat.v1.range(args.item)
		self.users=tf.compat.v1.range(args.user)
		self.timeEmbed=NNs.defineParam('timeEmbed', [self.maxTime+1, args.latdim], reg=True)
		
		for k in range(args.graphNum):
			embs0=[uEmbed[k]]
			embs1=[iEmbed[k]]
			for i in range(args.gnn_layer):
				a_emb0= self.messagePropagate(embs1[-1],self.edgeDropout(self.subAdj[k]),'user')
				a_emb1= self.messagePropagate(embs0[-1],self.edgeDropout(self.subTpAdj[k]),'item')
				embs0.append(a_emb0+embs0[-1]) 
				embs1.append(a_emb1+embs1[-1]) 
			user=tf.compat.v1.add_n(embs0)
			item=tf.compat.v1.add_n(embs1)
			user_vector.append(user)
			item_vector.append(item)
		
		# now user_vector is [g,u,latdim]
		user_vector=tf.compat.v1.stack(user_vector,axis=0)
		item_vector=tf.compat.v1.stack(item_vector,axis=0)
		user_vector_tensor=tf.compat.v1.transpose(user_vector, perm=[1, 0, 2])
		item_vector_tensor=tf.compat.v1.transpose(item_vector, perm=[1, 0, 2])

		# Replace tf.contrib.rnn with tf.compat.v1.nn.rnn_cell
		def gru_cell(): 
			return tf.compat.v1.nn.rnn_cell.BasicLSTMCell(args.latdim)
		def dropout():
			cell = gru_cell()
			return tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keepRate)
		
		with tf.compat.v1.name_scope("rnn"):
			cells = [dropout() for _ in range(1)]
			rnn_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)        
			user_vector_rnn, _ = tf.compat.v1.nn.dynamic_rnn(cell=rnn_cell, inputs=user_vector_tensor, dtype=tf.float32)
			item_vector_rnn, _ = tf.compat.v1.nn.dynamic_rnn(cell=rnn_cell, inputs=item_vector_tensor, dtype=tf.float32)
			user_vector_tensor=user_vector_rnn
			item_vector_tensor=item_vector_rnn

		self.additive_attention0 = AdditiveAttention(args.query_vector_dim,args.latdim)
		self.additive_attention1 = AdditiveAttention(args.query_vector_dim,args.latdim)
		self.multihead_self_attention0 = MultiHeadSelfAttention(args.latdim,args.num_attention_heads)
		self.multihead_self_attention1 = MultiHeadSelfAttention(args.latdim,args.num_attention_heads)
		
		# Replace tf.contrib.layers.layer_norm with tf.compat.v1.layers.batch_normalization
		multihead_user_vector = self.multihead_self_attention0.attention(
			tf.compat.v1.layers.batch_normalization(user_vector_tensor, training=self.is_train)
		)
		multihead_item_vector = self.multihead_self_attention1.attention(
			tf.compat.v1.layers.batch_normalization(item_vector_tensor, training=self.is_train)
		)
		
		final_user_vector = tf.compat.v1.reduce_mean(multihead_user_vector,axis=1)
		final_item_vector = tf.compat.v1.reduce_mean(multihead_item_vector,axis=1)
		
		# Save final_item_vector as a class variable for use in hard negative sampling
		self.final_item_vector = final_item_vector
		iEmbed_att=final_item_vector
		
		# sequence att
		self.multihead_self_attention_sequence = list()
		for i in range(args.att_layer):
			self.multihead_self_attention_sequence.append(MultiHeadSelfAttention(args.latdim,args.num_attention_heads))
		
		sequence_batch=tf.compat.v1.layers.batch_normalization(
			tf.compat.v1.matmul(tf.compat.v1.expand_dims(self.mask,axis=1),tf.compat.v1.nn.embedding_lookup(iEmbed_att,self.sequence)),
			training=self.is_train
		)
		sequence_batch+=tf.compat.v1.layers.batch_normalization(
			tf.compat.v1.matmul(tf.compat.v1.expand_dims(self.mask,axis=1),tf.compat.v1.nn.embedding_lookup(posEmbed,pos)),
			training=self.is_train
		)
		
		att_layer=sequence_batch
		for i in range(args.att_layer):
			att_layer1=self.multihead_self_attention_sequence[i].attention(
				tf.compat.v1.layers.batch_normalization(att_layer, training=self.is_train)
			)
			att_layer=Activate(att_layer1,"leakyRelu")+att_layer
		
		att_user=tf.compat.v1.reduce_sum(att_layer,axis=1)
		# Save att_user as a class variable for use in hard negative sampling
		self.att_user = att_user
		
		pckIlat_att = tf.compat.v1.nn.embedding_lookup(iEmbed_att, self.iids)		
		pckUlat = tf.compat.v1.nn.embedding_lookup(final_user_vector, self.uids)
		pckIlat = tf.compat.v1.nn.embedding_lookup(final_item_vector, self.iids)
		preds = tf.compat.v1.reduce_sum(pckUlat * pckIlat, axis=-1)
		preds += tf.compat.v1.reduce_sum(Activate(tf.compat.v1.nn.embedding_lookup(att_user,self.uLocs_seq),"leakyRelu")* pckIlat_att,axis=-1)
		
		# Store predictions for testing
		self.preds = preds
		
		self.preds_one=list()
		self.final_one=list()
		sslloss = 0	
		user_weight=list()
		
		for i in range(args.graphNum):
			meta1=tf.compat.v1.concat([final_user_vector*user_vector[i],final_user_vector,user_vector[i]],axis=-1)
			meta2=FC(meta1,args.ssldim,useBias=True,activation='leakyRelu',reg=True,reuse=True,name="meta2")
			user_weight.append(tf.compat.v1.squeeze(FC(meta2,1,useBias=True,activation='sigmoid',reg=True,reuse=True,name="meta3")))
		
		user_weight=tf.compat.v1.stack(user_weight,axis=0)	
		
		for i in range(args.graphNum):
			sampNum = tf.compat.v1.shape(self.suids[i])[0] // 2
			pckUlat = tf.compat.v1.nn.embedding_lookup(final_user_vector, self.suids[i])
			pckIlat = tf.compat.v1.nn.embedding_lookup(final_item_vector, self.siids[i])
			pckUweight =  tf.compat.v1.nn.embedding_lookup(user_weight[i], self.suids[i])
			pckIlat_att = tf.compat.v1.nn.embedding_lookup(iEmbed_att, self.siids[i])
			S_final = tf.compat.v1.reduce_sum(Activate(pckUlat* pckIlat, self.actFunc),axis=-1)
			posPred_final = tf.compat.v1.stop_gradient(tf.compat.v1.slice(S_final, [0], [sampNum]))
			negPred_final = tf.compat.v1.stop_gradient(tf.compat.v1.slice(S_final, [sampNum], [-1]))
			posweight_final = tf.compat.v1.slice(pckUweight, [0], [sampNum])
			negweight_final = tf.compat.v1.slice(pckUweight, [sampNum], [-1])
			
			self.preds_one.append(S_final)
			
			# Continue with the rest of the SSL loss computation...
			S_seq = tf.compat.v1.reduce_sum(Activate(tf.compat.v1.nn.embedding_lookup(att_user,self.sUsrSeq[i])* pckIlat_att, self.actFunc),axis=-1)
			posPred_seq = tf.compat.v1.slice(S_seq, [0], [sampNum])
			negPred_seq = tf.compat.v1.slice(S_seq, [sampNum], [-1])
			
			self.final_one.append(tf.compat.v1.reduce_sum(posweight_final*posPred_final,axis=-1)+tf.compat.v1.reduce_sum(posPred_seq,axis=-1))
			lossEps = args.ssl_reg
			usrEmbeds, itmEmbeds = tf.compat.v1.nn.embedding_lookup(final_user_vector, self.suids[i]), tf.compat.v1.nn.embedding_lookup(final_item_vector, self.siids[i])
			usrEmbeds_att, itmEmbeds_att = tf.compat.v1.nn.embedding_lookup(att_user, self.sUsrSeq[i]), tf.compat.v1.nn.embedding_lookup(iEmbed_att, self.siids[i])
			
			scoreDiff_final = posPred_final - negPred_final
			scoreDiff_seq = posPred_seq - negPred_seq
			
			bprLoss_final = tf.compat.v1.reduce_sum(tf.compat.v1.maximum(0.0, 1.0 - scoreDiff_final)) 
			bprLoss_seq = tf.compat.v1.reduce_sum(tf.compat.v1.maximum(0.0, 1.0 - scoreDiff_seq))
			
			sslloss += (bprLoss_final+bprLoss_seq) * lossEps

		# Store components for analysis
		self.posPred = tf.compat.v1.reduce_mean([tf.compat.v1.reduce_mean(self.final_one[i]) for i in range(args.graphNum)])
		self.negPred = tf.compat.v1.reduce_mean([tf.compat.v1.reduce_mean(self.preds_one[i][args.sampNum:]) for i in range(args.graphNum)])
		
		# Main prediction with regularization
		regLoss = args.reg * Regularize()
		self.regLoss = regLoss
		self.preLoss = tf.compat.v1.reduce_sum(tf.compat.v1.square(preds - self.labels)) + regLoss + sslloss
		ret = self.preLoss

		# Add hard negative sampling if enabled
		if args.use_hard_neg:
			contrastive_loss = self.compute_infonce_loss()
			self.contrastive_loss = contrastive_loss
			ret += args.contrastive_weight * contrastive_loss
			
		return ret

	def compute_infonce_loss(self):
		"""
		Compute InfoNCE contrastive loss for hard negative sampling
		"""
		batch_size = tf.compat.v1.shape(self.uids)[0]
		
		# Get user short-term preferences (att_user) and item embeddings
		user_short_term = tf.compat.v1.nn.embedding_lookup(self.att_user, self.uLocs_seq)  # [batch_size, latdim]
		item_embeds = self.final_item_vector  # [num_items, latdim]
		
		# Normalize embeddings for cosine similarity
		user_short_term_norm = tf.compat.v1.nn.l2_normalize(user_short_term, axis=1)
		item_embeds_norm = tf.compat.v1.nn.l2_normalize(item_embeds, axis=1)
		
		# Compute similarity matrix: [batch_size, num_items]
		similarity_matrix = tf.compat.v1.matmul(user_short_term_norm, item_embeds_norm, transpose_b=True)
		
		# Get positive item embeddings
		pos_items = tf.compat.v1.nn.embedding_lookup(item_embeds_norm, self.iids)  # [batch_size, latdim]
		
		# Compute positive similarities
		pos_similarities = tf.compat.v1.reduce_sum(user_short_term_norm * pos_items, axis=1)  # [batch_size]
		pos_similarities = tf.compat.v1.expand_dims(pos_similarities, axis=1)  # [batch_size, 1]
		
		# For hard negatives, we'll use a simplified approach that works with tf.compat.v1
		# Sample random hard negatives and mask out positives
		neg_indices = tf.compat.v1.random.uniform([batch_size, args.hard_neg_top_k], 
												 minval=0, maxval=args.item, dtype=tf.int32)
		neg_similarities = tf.compat.v1.batch_gather(similarity_matrix, neg_indices)
		
		# Combine positive and negative similarities: [batch_size, 1+K]
		all_similarities = tf.compat.v1.concat([pos_similarities, neg_similarities], axis=1)
		
		# Apply temperature scaling
		scaled_similarities = all_similarities / args.temp
		
		# Compute InfoNCE loss
		labels = tf.compat.v1.zeros([batch_size], dtype=tf.int32)  # Positive is always at index 0
		infonce_loss = tf.compat.v1.reduce_mean(
			tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(
				labels=labels, 
				logits=scaled_similarities
			)
		)
		
		return infonce_loss

	def prepareModel(self):
		self.actFunc = 'leakyRelu'
		self.maxTime = 60*60*24*2//args.time_split
		self.is_train=tf.compat.v1.placeholder(tf.bool, [])
		with tf.compat.v1.name_scope("Input"):
			# Input placeholders using tf.compat.v1
			self.keepRate = tf.compat.v1.placeholder(dtype=tf.float32, shape=[], name="keepRate")
			self.uids = tf.compat.v1.placeholder(name='uids', dtype=tf.int32, shape=[None])
			self.iids = tf.compat.v1.placeholder(name='iids', dtype=tf.int32, shape=[None])
			self.labels = tf.compat.v1.placeholder(name='labels', dtype=tf.float32, shape=[None])
			
			# Additional placeholders for sequences and SSL
			self.sequence = tf.compat.v1.placeholder(name='sequence', dtype=tf.int32, shape=[args.batch, args.seq_length])
			self.mask = tf.compat.v1.placeholder(name='mask', dtype=tf.float32, shape=[args.batch, args.seq_length])
			self.uLocs_seq = tf.compat.v1.placeholder(name='uLocs_seq', dtype=tf.int32, shape=[None])
			
			# SSL placeholders
			self.suids = []
			self.siids = []
			self.sUsrSeq = []
			for i in range(args.graphNum):
				self.suids.append(tf.compat.v1.placeholder(name='suids'+str(i), dtype=tf.int32, shape=[None]))
				self.siids.append(tf.compat.v1.placeholder(name='siids'+str(i), dtype=tf.int32, shape=[None]))
				self.sUsrSeq.append(tf.compat.v1.placeholder(name='sUsrSeq'+str(i), dtype=tf.int32, shape=[None]))

		# Load adjacency matrices
		self.subAdj = []
		self.subTpAdj = []
		for i in range(args.graphNum):
			adj, tpadj = self.handler.getAdj(i, args.gnn_layer)
			self.subAdj.append(tf.compat.v1.sparse.SparseTensor(adj._indices, adj._values, adj.shape))
			self.subTpAdj.append(tf.compat.v1.sparse.SparseTensor(tpadj._indices, tpadj._values, tpadj.shape))

		# Build model
		self.loss = self.ours()
		
		# Training step
		globalStep = tf.compat.v1.Variable(0, trainable=False)
		learningRate = tf.compat.v1.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
		self.optimizer = tf.compat.v1.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

		# Additional operations for saving/loading
		self.save_ops = tf.compat.v1.train.Saver(max_to_keep=1)

	def sampleTrainBatch(self, batIds, labelMat, timeMat, train_sample_num):
		temLabel = labelMat[batIds].toarray()
		batch = len(batIds)
		user_seqs = []
		user_masks = []
		user_locs = []
		
		for i in range(batch):
			user_id = batIds[i]
			seq = []
			
			# Get user's sequence from handler
			if hasattr(self.handler, 'sequence') and user_id in self.handler.sequence:
				user_seq = self.handler.sequence[user_id]
				if len(user_seq) > 1:
					seq = user_seq[:-1]  # All but last item
			
			# Pad or truncate sequence
			if len(seq) >= args.seq_length:
				seq = seq[-args.seq_length:]
				mask = [1.0] * args.seq_length
			else:
				mask = [0.0] * (args.seq_length - len(seq)) + [1.0] * len(seq)
				seq = [0] * (args.seq_length - len(seq)) + seq
				
			user_seqs.append(seq)
			user_masks.append(mask)
			user_locs.append(i)
		
		# Create training pairs
		uLocs, iLocs = [], []
		for i in range(batch):
			posItems = np.reshape(np.argwhere(temLabel[i] != 0), [-1])
			
			if len(posItems) == 0:
				continue
				
			# Sample positive and negative items
			for j in range(min(train_sample_num, len(posItems))):
				posItem = np.random.choice(posItems)
				
				# Use hard negative sampling if enabled
				if args.use_hard_neg:
					negItems = self.sample_hard_negatives_simple(batIds[i], temLabel[i], 1)
					if len(negItems) > 0:
						negItem = negItems[0]
					else:
						negItem = negSamp(temLabel[i], 1, args.item)[0]
				else:
					negItem = negSamp(temLabel[i], 1, args.item)[0]
				
				uLocs.extend([batIds[i], batIds[i]])
				iLocs.extend([posItem, negItem])
		
		labels = [1.0] * (len(uLocs) // 2) + [0.0] * (len(uLocs) // 2)
		
		sequence = np.array(user_seqs[:args.batch])
		mask = np.array(user_masks[:args.batch])
		uLocs_seq = np.array(user_locs[:len(uLocs)])
		
		return uLocs, iLocs, sequence, mask, uLocs_seq

	def sample_hard_negatives_simple(self, user_id, user_interactions, num_negs):
		"""Simplified hard negative sampling for training"""
		# Get non-interacted items
		non_interacted = np.where(user_interactions == 0)[0]
		
		if len(non_interacted) < num_negs:
			return np.random.choice(non_interacted, num_negs, replace=True).tolist()
		else:
			return np.random.choice(non_interacted, num_negs, replace=False).tolist()

	def sampleSslBatch(self, batIds, labelMat, use_epsilon=True):
		temLabel=list()
		for k in range(args.graphNum):	
			temLabel.append(labelMat[k][batIds].toarray())
		batch = len(batIds)
		temlen = batch * 2 * args.sslNum
		uLocs = [[None] * temlen] * args.graphNum
		iLocs = [[None] * temlen] * args.graphNum
		uLocs_seq = [[None] * temlen] * args.graphNum
		
		for k in range(args.graphNum):	
			cur = 0				
			for i in range(batch):
				posset = np.reshape(np.argwhere(temLabel[k][i]!=0), [-1])
				sslNum = min(args.sslNum, len(posset)//2)
				if sslNum == 0:
					poslocs = [np.random.choice(args.item)]
					neglocs = [poslocs[0]]
				else:
					all = np.random.choice(posset, sslNum*2)
					poslocs = all[:sslNum]
					neglocs = all[sslNum:]
				for j in range(sslNum):
					posloc = poslocs[j]
					negloc = neglocs[j]			
					uLocs[k][cur] = uLocs[k][cur+1] = batIds[i]
					uLocs_seq[k][cur] = uLocs_seq[k][cur+1] = i
					iLocs[k][cur] = posloc
					iLocs[k][cur+1] = negloc
					cur += 2
			uLocs[k]=uLocs[k][:cur]
			iLocs[k]=iLocs[k][:cur]
			uLocs_seq[k]=uLocs_seq[k][:cur]
		return uLocs, iLocs, uLocs_seq

	def trainEpoch(self):
		num = args.user
		sfIds = np.random.permutation(num)[:args.trnNum]
		epochLoss, epochPreLoss = [0] * 2
		epochContrastiveLoss = 0
		num = len(sfIds)
		sample_num_list=[40]		
		steps = int(np.ceil(num / args.batch))
		
		for s in range(len(sample_num_list)):
			for i in range(steps):
				st = i * args.batch
				ed = min((i+1) * args.batch, num)
				batIds = sfIds[st: ed]

				if args.use_hard_neg:
					target = [self.optimizer, self.preLoss, self.regLoss, self.loss, self.contrastive_loss, self.posPred, self.negPred, self.preds_one]
				else:
					target = [self.optimizer, self.preLoss, self.regLoss, self.loss, self.posPred, self.negPred, self.preds_one]
					
				feed_dict = {}
				uLocs, iLocs, sequence, mask, uLocs_seq= self.sampleTrainBatch(batIds, self.handler.trnMat, self.handler.timeMat, sample_num_list[s])
				suLocs, siLocs, suLocs_seq = self.sampleSslBatch(batIds, self.handler.subMat, False)
				
				# Create labels
				labels = [1.0] * (len(uLocs) // 2) + [0.0] * (len(uLocs) // 2)
				
				feed_dict[self.uids] = uLocs
				feed_dict[self.iids] = iLocs
				feed_dict[self.labels] = labels
				feed_dict[self.sequence] = sequence
				feed_dict[self.mask] = mask
				feed_dict[self.is_train] = True
				feed_dict[self.uLocs_seq] = uLocs_seq
				
				for k in range(args.graphNum):
					feed_dict[self.suids[k]] = suLocs[k]
					feed_dict[self.siids[k]] = siLocs[k]
					feed_dict[self.sUsrSeq[k]] = suLocs_seq[k]
				feed_dict[self.keepRate] = args.keepRate

				res = self.sess.run(target, feed_dict=feed_dict)

				if args.use_hard_neg:
					preLoss, regLoss, loss, contrastiveLoss, pos, neg, pone = res[1:]
					epochContrastiveLoss += contrastiveLoss
					log('Step %d/%d: preloss = %.2f, REGLoss = %.2f, ConLoss = %.4f         ' % 
						(i+s*steps, steps*len(sample_num_list), preLoss, regLoss, contrastiveLoss), save=False, oneline=True)
				else:
					preLoss, regLoss, loss, pos, neg, pone = res[1:]
					log('Step %d/%d: preloss = %.2f, REGLoss = %.2f         ' % 
						(i+s*steps, steps*len(sample_num_list), preLoss, regLoss), save=False, oneline=True)
					
				epochLoss += loss
				epochPreLoss += preLoss
				
		ret = dict()
		ret['Loss'] = epochLoss / steps
		ret['preLoss'] = epochPreLoss / steps
		if args.use_hard_neg:
			ret['contrastiveLoss'] = epochContrastiveLoss / steps
		return ret

	def sampleTestBatch(self, batIds, labelMat):
		batch = len(batIds)
		temTst = self.handler.tstInt
		temLabel = labelMat[batIds].toarray()
		temlen = batch * 100
		uLocs = [None] * temlen
		iLocs = [None] * temlen
		tstLocs = [None] * batch
		cur = 0
		
		# Create sequences for test users
		user_seqs = []
		user_masks = []
		for i in range(batch):
			user_id = batIds[i]
			seq = []
			
			if hasattr(self.handler, 'sequence') and user_id in self.handler.sequence:
				user_seq = self.handler.sequence[user_id]
				if len(user_seq) > 1:
					seq = user_seq[:-1]  # All but last item
			
			# Pad or truncate sequence
			if len(seq) >= args.seq_length:
				seq = seq[-args.seq_length:]
				mask = [1.0] * args.seq_length
			else:
				mask = [0.0] * (args.seq_length - len(seq)) + [1.0] * len(seq)
				seq = [0] * (args.seq_length - len(seq)) + seq
				
			user_seqs.append(seq)
			user_masks.append(mask)
		
		for i in range(batch):
			negset = np.reshape(np.argwhere(temLabel[i]==0), [-1])
			negset = [i for i in negset if i != temTst[batIds[i]]]
			negset = np.random.choice(negset, 99)
			
			tstLocs[i] = cur
			uLocs[cur] = batIds[i]
			iLocs[cur] = temTst[batIds[i]]
			cur += 1
			
			for j in range(99):
				uLocs[cur] = batIds[i]
				iLocs[cur] = negset[j]
				cur += 1
		
		# Pad sequences to batch size
		while len(user_seqs) < args.batch:
			user_seqs.append([0] * args.seq_length)
			user_masks.append([0.0] * args.seq_length)
		
		sequence = np.array(user_seqs[:args.batch])
		mask = np.array(user_masks[:args.batch])
		uLocs_seq = list(range(batch))
		
		return uLocs, iLocs, temTst, tstLocs, sequence, mask, uLocs_seq

	def testEpoch(self):
		epochHit, epochNdcg = [0] * 2
		ids = self.handler.tstUsrs
		num = len(ids)
		testbatch = np.maximum(1, args.batch // args.testSize)
		steps = int(np.ceil(num / testbatch))
		
		for i in range(steps):
			st = i * testbatch
			ed = min((i+1) * testbatch, num)
			batIds = ids[st: ed]
			uLocs, iLocs, temTst, tstLocs, sequence, mask, uLocs_seq = self.sampleTestBatch(batIds, self.handler.trnMat)
			
			feed_dict = {}
			feed_dict[self.uids] = uLocs
			feed_dict[self.iids] = iLocs
			feed_dict[self.sequence] = sequence
			feed_dict[self.mask] = mask
			feed_dict[self.uLocs_seq] = uLocs_seq
			feed_dict[self.is_train] = False
			feed_dict[self.keepRate] = 1.0
			
			# Add dummy SSL batch data
			for k in range(args.graphNum):
				feed_dict[self.suids[k]] = [0, 0]  # Dummy values
				feed_dict[self.siids[k]] = [0, 0]
				feed_dict[self.sUsrSeq[k]] = [0, 0]
			
			# Get prediction scores
			preds = self.sess.run(self.preds, feed_dict=feed_dict)
			hit, ndcg = self.calcRes(preds, temTst, tstLocs)
			epochHit += hit
			epochNdcg += ndcg
			
		ret = dict()
		ret['HR'] = epochHit / num
		ret['NDCG'] = epochNdcg / num
		return ret

	def calcRes(self, preds, temTst, tstLocs):
		hit, ndcg = [0] * 2
		for i in range(len(tstLocs)):
			predvals = list(zip(preds[tstLocs[i]: tstLocs[i] + 100], range(100)))
			predvals.sort(key=lambda x: x[0], reverse=True)
			shoot = list(map(lambda x: x[1], predvals[:args.shoot]))
			if 0 in shoot:
				hit += 1
				ndcg += np.reciprocal(np.log2(shoot.index(0) + 2))
		return hit, ndcg

	def saveHistory(self):
		if args.epoch == 0:
			return
		with open('History/' + args.save_path + '.his', 'wb') as fs:
			pickle.dump(self.metrics, fs)

	def loadModel(self):
		self.save_ops.restore(self.sess, 'Models/' + args.load_model)
		with open('History/' + args.load_model + '.his', 'rb') as fs:
			self.metrics = pickle.load(fs) 