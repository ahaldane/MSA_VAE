#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.stats import norm, pearsonr
from pathlib import Path
import argparse, sys, time, pickle

from seqload import loadSeqs, writeSeqs
from seqtools import histsim


import keras
from keras.models import Model
from keras import regularizers
from keras.layers import Input, Dense, Lambda, Dropout, Activation
from keras import backend as K
from keras.objectives import categorical_crossentropy
from keras.callbacks import EarlyStopping, CSVLogger
from keras.layers.normalization import BatchNormalization

#ALPHA="XILVAGMFYWEDQNHCRKSTPBZ-"[::-1]
ALPHA = "-ACDEFGHIKLMNPQRSTVWY"
q = len(ALPHA)

class OneHot_Generator(keras.utils.Sequence) :
    """
    Keras data generator which converts sequences to one-hot representation
    in batches.
    """
    def __init__(self, seqs, batch_size) :
        self.seqs = seqs
        self.batch_size = batch_size
        assert(seqs.shape[0] % batch_size == 0)

    def __len__(self) :
        return self.seqs.shape[0] // self.batch_size

    def __getitem__(self, idx) :
        N = self.batch_size
        batch = self.seqs[idx*N:idx*N + N]

        L = self.seqs.shape[1]
        one_hot = np.zeros((N, L, q), dtype='float32')
        one_hot[np.arange(N)[:,None], np.arange(L)[None,:], batch] = 1
        one_hot = one_hot.reshape((N, L*q))
        return one_hot, one_hot

class TVD_Evaluation(keras.callbacks.Callback):
    def __init__(self, vae, ref_seqs):
        super().__init__()
        self.vae = vae

        self.h = histsim(ref_seqs).astype(float)
        self.h = self.h/np.sum(self.h)

    def on_train_begin(self, logs={}):
        self.TVDs = []

    def on_epoch_end(self, epoch, logs={}):
        seqs = np.concatenate(list(self.vae.generate(1000)))
        print('Hamming computation...')
        h = histsim(seqs).astype(float)
        h = h/np.sum(h)
        self.TVDs.append(np.sum(np.abs(self.h - h))/2)
        print("TVD:", self.TVDs[-1])

class Base_VAE:
    def __init__(self):
        self._TVDval = None

    def create_model(self, L, q, latent_dim, batch_size, *args):
        self.batch_size, self.latent_dim = batch_size, latent_dim
        self.L, self.q = L, q
        Lq = L*q

        encl, decl = self._enc_dec_layers(L, q, latent_dim, batch_size, *args)

        # set up encoder
        enc_in = h = Input(batch_shape=(batch_size, Lq), name='encoder_in')
        for layer in encl:
            h = layer(h)
        z_mean = Dense(latent_dim, name='z_mean')(h)
        z_log_var = Dense(latent_dim, name='z_log_var')(h)
        encoder = Model(enc_in, [z_mean, z_log_var], name='encoder')

        # MC sampling layer
        sample = Lambda(self._sampling, output_shape=(latent_dim,))

        # set up decoder
        dec_in = d = Input((latent_dim,), name='decoder_in')
        for layer in decl:
            d = layer(d)
        d = Dense(Lq, activation='sigmoid')(d)
        decoder = Model(dec_in, d, name='decoder')

        self.vae = Model(enc_in, decoder(sample(encoder(enc_in))))

        self._extract_layers()

    def _enc_dec_layers(self, L, q, latent_dim, batch_size, *args):
        raise NotImplementedError()

    def summarize(self):
        print("**************************************")
        self.encoder.summary()
        print("**************************************")
        self.decoder.summary()
        print("**************************************")
        self.vae.summary()

    def _extract_layers(self):
        # extract important layers from the vae and hold on to them
        self.encoder = self.vae.get_layer('encoder')
        self.decoder = self.vae.get_layer('decoder')

        self.z_mean = self.encoder.get_layer('z_mean').output
        self.z_log_var = self.encoder.get_layer('z_log_var').output

    def save(self, name):
        with open('{}_param.pkl'.format(name), 'wb') as f:
            d = (self.batch_size, self.L, self.q, self.latent_dim, 
                 self.__class__.__name__)
            pickle.dump(d, f)
        self.vae.save(name + '_vae.k')

    def load(self, name):
        with open('{}_param.pkl'.format(name), 'rb') as f:
            d = pickle.load(f)
            self.batch_size, self.L, self.q, self.latent_dim, cls = d
        self.vae = keras.models.load_model(name + '_vae.k', compile=False,
                                   custom_objects={'_sampling': self._sampling,
                                                   '_vae_loss': self._vae_loss})
        self._extract_layers()
        self.vae.compile(optimizer="adam", loss=self._vae_loss)

    def _sampling(self, args):
        zm, zlv = args
        epsilon = K.random_normal((self.batch_size, self.latent_dim), 0., 1.)
        return zm + K.exp(zlv / 2) * epsilon

    def _vae_loss(self, x, x_decoded_mean, **kwarg):
        zm, zlv = self.z_mean, self.z_log_var
        # Original Church code has a prefactor of Lq, but it seems wrong...
        # It may have been copied from the fchollet example code, also wrong
        #Lq = self.L*self.q
        #xent_loss = Lq * categorical_crossentropy(x,  x_decoded_mean)
        xent_loss = categorical_crossentropy(x, x_decoded_mean)
        kl_loss = 0.5*K.sum(1 + zlv - K.square(zm) - K.exp(zlv), axis=-1)
        return xent_loss - kl_loss

    def getTVDlog(self):
        return None if self._TVDval is None else self._TVDval.TVDs

    def train(self, epochs, train_seq, validation_seq, name=None, TVDseqs=None):
        # Comment from Church:
        # Potentially better results, but requires further hyperparameter tuning
        #optimizer = keras.optimizers.SGD(lr=0.005, momentum=0.001, decay=0.0,
        #                                 nesterov=False, clipvalue=0.05)
        self.vae.compile(optimizer="adam", loss=self._vae_loss)

        x_train = OneHot_Generator(train_seq, self.batch_size)
        x_valid = OneHot_Generator(validation_seq, self.batch_size)

        callbacks = [EarlyStopping(monitor='val_loss', patience=3), ]
        if name is not None:
            callbacks.append(CSVLogger("{}_train_log.csv".format(name)))
        if TVDseqs is not None:
            self._TVDval = TVD_Evaluation(self, TVDseqs)
            callbacks.append(self._TVDval)

        hist = self.vae.fit(x_train,
                            shuffle=True,
                            epochs=epochs,
                            #sample_weight=np.array(new_weights),
                            validation_data=x_valid,
                            callbacks=callbacks)
        return hist

    def encode(self, data):
        return self.encoder.predict(OneHot_Generator(data, self.batch_size))

    def decode_bernoulli(self, z):
        brnll = self.decoder.predict(z)
        brnll = brnll.reshape((z.shape[0], self.L, self.q))
        # clip like in Keras categorical_crossentropy used in vae_loss
        brnll = np.clip(brnll, 1e-7, 1 - 1e-7)
        brnll = brnll/np.sum(brnll, axis=-1, keepdims=True)
        return brnll

    def single_sample(self, data):
        return self.vae.predict(OneHot_Generator(data, self.batch_size))

    def lELBO(self, seqs, n_samples=1000):
        N, L = seqs.shape
        rN, rL = np.arange(N)[:,None], np.arange(L)

        zm, zlv = self.encode(seqs)
        zstd = np.exp(zlv/2)

        kl_loss = 0.5*np.sum(1 + zlv - np.square(zm) - np.exp(zlv), axis=-1)

        xent_loss = np.zeros(N, dtype=float)
        for n in range(n_samples):
            z = norm.rvs(zm, zstd)
            brnll = self.decode_bernoulli(z)
            xent_loss += np.sum(-np.log(brnll[rN, rL, seqs]), axis=-1)
        xent_loss /= n_samples

        return xent_loss - kl_loss

    def logp(self, seqs, n_samples=1000):
        N, L = seqs.shape
        rN, rL = np.arange(N)[:,None], np.arange(L)

        zm, zlv = self.encode(seqs)
        zstd = np.exp(zlv/2)

        logp = None
        for n in range(n_samples):
            z = norm.rvs(zm, zstd)
            brnll = self.decode_bernoulli(z)

            lqz_x = np.sum(norm.logpdf(z, zm, zstd), axis=-1)
            lpx_z = np.sum(np.log(brnll[rN, rL, seqs]), axis=-1)
            lpz = np.sum(norm.logpdf(z, 0, 1), axis=-1)
            lpxz = lpz + lpx_z

            if logp is None:
                logp = lpxz - lqz_x
            else:
                np.logaddexp(logp, lpxz - lqz_x, out=logp)

        return logp - np.log(n_samples)

    def generate(self, N):
        # returns a generator yielding sequences in batches
        assert(N % self.batch_size == 0)

        print("")
        for n in range(N // self.batch_size):
            print("\rGen {}/{}".format(n*self.batch_size, N), end='')

            z = norm.rvs(0., 1., size=(self.batch_size, self.latent_dim))
            brnll = self.decode_bernoulli(z)

            c = np.cumsum(brnll, axis=2)
            c = c/c[:,:,-1,None] # correct for fp error
            r = np.random.rand(self.batch_size, self.L)

            seqs = np.sum(r[:,:,None] > c, axis=2, dtype='u1')
            yield seqs
        print("\rGen {}/{}   ".format(N, N))

class Church_VAE(Base_VAE):
    def __init__(self):
        super().__init__()

    def _enc_dec_layers(self, L, q, latent_dim, batch_size, inner_dim):
        inner_dim = int(inner_dim)

        enc_layers = [Dense(inner_dim, activation="elu"),
                      Dropout(0.3),
                      Dense(inner_dim, activation='elu'),
                      BatchNormalization(),
                      Dense(inner_dim, activation='elu')]

        dec_layers = [Dense(inner_dim, activation='elu'),
                      Dense(inner_dim, activation='elu'),
                      Dropout(0.3),
                      Dense(inner_dim, activation='elu')]

        return enc_layers, dec_layers

class Deep_VAE(Base_VAE):
    def __init__(self):
        super().__init__()

    def _enc_dec_layers(self, L, q, latent_dim, batch_size, depth):
        self.batch_size, self.latent_dim = batch_size, latent_dim
        Lq = L*q
        depth = int(depth)

        enc_layers = [Dense(Lq//(2**(d + 1)), activation='elu') 
                      for d in range(depth)]
        dec_layers = [Dense(Lq//(2**(d + 1)), activation='elu') 
                       for d in range(depth)[::-1]]

        return enc_layers, dec_layers

class VVAE(Base_VAE):
    def __init__(self):
        super().__init__()

    def _enc_dec_layers(self, L, q, latent_dim, batch_size):
        Lq = L*q

        enc_layers = [Dense(Lq//2, activation="elu"),
                      #Dropout(0.3),
                      Dense(Lq//4, activation='elu'),
                      BatchNormalization(),
                      Dense(Lq//8, activation='elu')]

        dec_layers = [Dense(Lq//8, activation='elu'),
                      Dense(Lq//4, activation='elu'),
                      #Dropout(0.3),
                      Dense(Lq//2, activation='elu')]

        return enc_layers, dec_layers

vaes = {'VVAE': VVAE, 'Church_VAE': Church_VAE, 'Deep_VAE': Deep_VAE}

def loadVAE(name):
    with open('{}_param.pkl'.format(name), 'rb') as f:
        vtype = pickle.load(f)[-1]
    v = vaes[vtype]()
    v.load(name)
    return v

def plot_performance(vae, hist, name):
    #print performance measures over time
    for key in ["loss"]:
        plt.figure()
        plt.title(key)
        plt.plot(hist.history[key], label=key)
        plt.plot(hist.history["val_"+key], label="val_"+key)
        plt.xlabel("epochs")
        plt.legend();
        plt.savefig('Training_{}_{}.png'.format(key, name))
        plt.close()

    tvdlog = vae.getTVDlog()
    if tvdlog is not None:
        plt.figure()
        plt.title('TVD')
        plt.plot(tvdlog, label=key)
        plt.xlabel("epochs")
        plt.legend();
        plt.savefig('Training_{}_{}.png'.format("TVD", name))
        plt.close()

def main_plot_latent(name, args):
    parser = argparse.ArgumentParser()
    parser.add_argument('seqs')
    args = parser.parse_args(args)

    seqs = loadSeqs(args.seqs, alpha=ALPHA)[0]
    # only plot first 10,000
    seqs = seqs[:10000]

    vae = loadVAE(name)
    latent_dim = vae.latent_dim

    m, lv = vae.encode(seqs)
    st = np.exp(lv/2)
    
    # make 1d distribution plots
    fig = plt.figure(figsize=(12,12))
    nx = max(latent_dim//2, 1)
    ny = (latent_dim-1)//nx + 1
    for z1 in range(latent_dim):
        fig.add_subplot(nx,ny,z1+1)
        h, b, _ = plt.hist(m[:,z1], bins=100, density=True)
        wm, ws = m[0][z1], st[0][z1]
        x = np.linspace(wm - 5*ws, wm + 5*ws, 200)
        y = norm.pdf(x, wm, ws)
        y = y*np.max(h)/np.max(y)
        plt.plot(x, y, 'r-')
        plt.xlim(-5,5)
        plt.title('Z{}, <z{}>_std={:.2f}'.format(z1, z1, np.std(m[:,z1])))

    plt.savefig('LatentTraining_1d_{}.png'.format(name))
    plt.close()

    # make 2d distribution plots
    s = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(s, s)
    red = np.broadcast_to(np.array([1.,0, 0, 1]), (len(s), len(s), 4)).copy()

    fig = plt.figure(figsize=(12,12))
    counter = 1
    for z1 in range(latent_dim):
        print('Var z{}: {}'.format(z1, np.var(m[:, z1])))
        for z2 in range(z1+1,latent_dim):
            counter+=1
            fig.add_subplot(latent_dim,latent_dim,counter)

            plt.scatter(m[:, z1], m[:, z2], c='b', alpha=0.01)
            nn = (norm.pdf(X, m[0][z1], st[0][z1]) *
                  norm.pdf(Y, m[0][z2], st[0][z2]))
            nn = nn/np.max(nn)/1.5
            red[:,:,3] = nn
            plt.imshow(red, extent=(-5,5,-5,5), origin='lower')

            ##wildtype in red
            #plt.scatter(m[0][z1], m[0][z2],c="r", alpha=1)
            # make 1std oval for wt
            wtv = Ellipse((m[0][z1],  m[0][z2]),
                          width=2*st[0][z1], height=2*st[0][z2],
                          facecolor='none', edgecolor='red')
            plt.gca().add_patch(wtv)
            plt.xlim(-5,5)
            plt.ylim(-5,5)

            if z1 == 0:
                plt.xlabel('z{}'.format(z2))
                plt.gca().xaxis.set_label_position('top')
            if z2 == latent_dim -1:
                plt.ylabel('z{}'.format(z1))
                plt.gca().yaxis.set_label_position('right')
        counter += z1+2

    plt.savefig('LatentTraining_{}.png'.format(name))
    plt.close()

    # special plot for l=1 case: vary the 1 dimension, make movie of output
    if vae.latent_dim == 1:
        z = np.linspace(-4,4,vae.batch_size)
        psm = vae.decode_bernoulli(z)

        import matplotlib.animation as animation
        fig = plt.figure(figsize=(16,4))
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        h, b, _ = ax1.hist(m[:,0], bins=100, density=True)
        ax1.set_xlim(-4,4)
        artists = []
        for p, zi in zip(psm, z):
            zpt = ax1.plot([zi], [0], 'r.', ms=20)[0]
            im = ax2.imshow(p.T, cmap='gray_r',
                            interpolation='nearest', animated=True)
            artists.append([im, zpt])
            print("".join(ALPHA[c] for c in np.argmax(p, axis=1)))
        ani = animation.ArtistAnimation(fig, artists, interval=40, blit=True,
                                        repeat_delay=1000)
        #ani.save('vary_z1_{}.mp4'.format(name))
        ani.save('vary_z1_{}.gif'.format(name), dpi=80, writer='imagemagick')
        plt.close()

def main_seq_accuracy(name, args):
    parser = argparse.ArgumentParser()
    parser.add_argument('seqs')
    args = parser.parse_args(args)

    seqs = loadSeqs(args.seqs, alpha=ALPHA)[0]
    N, L = seqs.shape

    vae = loadVAE(name)

    # pad sequences to batch_size
    padN = ((N-1)//vae.batch_size + 1)*vae.batch_size
    padseqs = np.tile(seqs, ((padN-1)//N + 1, 1))[:padN]

    brnll = vae.single_sample(padseqs)
    brnll = brnll.reshape((brnll.shape[0], L, q))[:N,:,:]
    pwm = brnll/np.sum(brnll, axis=-1, keepdims=True)

    for n,s,o in zip(range(N), seqs, pwm):
        o = o.reshape(L,q)
        acc = np.mean(s == np.argmax(o, axis=1))
        p = np.sum(np.log(o[np.arange(L), s]))

        plt.figure(figsize=(16,4))
        plt.imshow(o.T, cmap='gray_r', interpolation='nearest')
        plt.plot(np.arange(L), s, 'r.', ms=2)
        plt.title("Seq {}   Acc={:.3f} log-p={:.4f}".format(str(n), acc, p))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.savefig("test_seq_{}_{}.png".format(name, n))
        plt.close()

def main_energy(name, args):
    parser = argparse.ArgumentParser()
    parser.add_argument('seqs')
    parser.add_argument('--ref_energy')
    parser.add_argument('--n_samples', default=1000, type=int)
    args = parser.parse_args(args)

    seqs = loadSeqs(args.seqs, alpha=ALPHA)[0]

    vae = loadVAE(name)

    nlelbo = -vae.lELBO(seqs, n_samples=args.n_samples)
    logp = vae.logp(seqs, n_samples=args.n_samples)

    np.save('nlelbo_{}', nlelbo)
    np.save('logp_{}', logp)

    plt.figure()
    plt.plot(nlelbo, logp, '.')
    plt.xlabel('$-\log$ ELBO')
    plt.ylabel('$\log p(x)$')
    plt.savefig("energies_{}.png".format(name))
    plt.close()

    if args.ref_energy:
        refE = np.load(args.ref_energy)

        plt.figure()
        plt.plot(refE, nlelbo, '.')
        plt.xlabel('Ref E')
        plt.xlabel('$-\log$ ELBO')
        plt.title(r'$\rho = {:.3f}$'.format(pearsonr(refE, nlelbo)[0]))
        plt.savefig("energies_elbo_{}.png".format(name))
        plt.close()

        plt.figure()
        plt.plot(refE, logp, '.')
        plt.xlabel('Ref E')
        plt.ylabel('$\log p(x)$')
        plt.title(r'$\rho = {:.3f}$'.format(pearsonr(refE, logp)[0]))
        plt.savefig("energies_logp_{}.png".format(name))
        plt.close()

def main_generate(name, args):
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int)
    args = parser.parse_args(args)
    N = args.N

    vae = loadVAE(name)

    with open('gen_{}_{}'.format(name, N), 'wb') as f:
        for seqs in vae.generate(N):
            writeSeqs(f, seqs, alpha=ALPHA)

def main_train(name, args):
    parser = argparse.ArgumentParser()
    parser.add_argument('vae_type', choices=vaes.keys())
    parser.add_argument('seqs')
    parser.add_argument('latent_dim', type=int)
    parser.add_argument('args', nargs='*')
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--TVDseqs', action='store_true')
    args = parser.parse_args(args)

    latent_dim = args.latent_dim
    seqs = loadSeqs(args.seqs, alpha=ALPHA)[0]
    N, L = seqs.shape
    print("L = {}".format(L))

    np.random.seed(42)
    batch_size = args.batch_size
    #inner_dim = args.inner_dim

    assert(N%batch_size == 0)
    n_batches = N//batch_size
    validation_batches = int(n_batches*0.1)
    train_seqs = seqs[:-validation_batches*batch_size]
    val_seqs = seqs[-validation_batches*batch_size:]
    TVDseqs = None
    if args.TVDseqs:
        TVDseqs = val_seqs[:1000]

    vae = vaes[args.vae_type]()
    vae.create_model(L, q, latent_dim, batch_size, *args.args)
    vae.summarize()
    hist = vae.train(60, train_seqs, val_seqs, name=name, TVDseqs=TVDseqs)
    vae.save(name)
    plot_performance(vae, hist, name)

def main_TVD(name, args):
    parser = argparse.ArgumentParser()
    parser.add_argument('ref_seqs')
    args = parser.parse_args(args)


    ref_seqs = loadSeqs(args.ref_seqs, alpha=ALPHA)[0]

    vae = loadVAE(name)

    rh = histsim(ref_seqs).astype(float)
    rh = rh/np.sum(rh)

    seqs = np.concatenate(list(vae.generate(10000)))
    h = histsim(seqs).astype(float)
    h = h/np.sum(h)

    plt.figure()
    plt.plot(rh, label='ref')
    plt.plot(h, label='model')
    plt.legend()
    plt.savefig("TVD_{}.png".format(name))

def main_summarize(name, args):
    vae = loadVAE(name)
    vae.summarize()

def main():
    funcs = {'train': main_train,
             'plot_latent': main_plot_latent,
             'seq_accuracy': main_seq_accuracy,
             'TVD': main_TVD,
             'energy': main_energy,
             'summarize': main_summarize,
             'gen': main_generate}

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('model_name')
    parser.add_argument('action', choices=funcs.keys())

    known_args, remaining_args = parser.parse_known_args(sys.argv[1:])


    funcs[known_args.action](known_args.model_name, remaining_args)

if __name__ == '__main__':
    main()
