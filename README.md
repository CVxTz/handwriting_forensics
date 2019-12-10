# handwriting_forensics

Data in : http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database/download-the-iam-on-line-handwriting-database

run.sh for training steps

### End-To-End Writer Identification : Off-line Vs On-line Approach


### Motivation :

Each person has their own distinct handwriting with its own special
characteristics and small details that makes it different. If we pick two
handwriting samples at random written by two people we might notice that the way
the pen strokes are constructed, the spacing between letters and other details
are different. In this post we will train a model that tries to classify if two
sample handwritings are written by the same person or not.

### Dataset :

We will use the [IAM On-Line Handwriting
Database](http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database/download-the-iam-on-line-handwriting-database)
which gives multiple handwriting samples per writer along with the sequence of
pen strokes that were recorded in the construction of the handwriting sample.

This dataset gives us the opportunity to tackle the problem of writer
identification using two separate approaches :

#### Off-line setting :

![](https://cdn-images-1.medium.com/max/800/1*3TlslcG3DrQoUsHwb2i9UQ.png)

<span class="figcaption_hack">Static representation of handwriting</span>


In the off-line setting we have a static image of the final result of the
handwriting. This gives a global representation of what the handwriting looks
like. This type of representation is probably best suited to be used as an input
to a vision model.

In this setting we will make two images like the one above to a siamese Vision
model ( Resnet50 in our case) and try to predict if the two samples were written
by the same person.

![](https://cdn-images-1.medium.com/max/800/1*GmfGe5bTCzE0k2bKkLd5Tg.png)

<span class="figcaption_hack">Simplified illustration of the off-line model</span>


The images are pre-processed to remove the extra white space on the edges and to
normalize their size to (128, 1024), while being careful to keep the same aspect
ratio before and after as to not deform the handwriting.

The same encoder is used for each image to produce a fixed length representation
for each of them, then the two representations are combined and fed to fully
connected layer, to finally make the prediction.

Since we have a set of handwritings made by each writer, we will use this
information to create positive samples by selecting two examples from the same
writer and negative examples by selecting samples from two different writers. We
also make sure that the same writers cannot be in the training set and the
evaluation set at the same time.

#### On-Line setting :

![](https://cdn-images-1.medium.com/max/800/1*sBnVwZ1OUY73Vq_fK_x2dQ.gif)

<span class="figcaption_hack">Handwriting with individual strokes details</span>


Since we also have the handwriting represented as a sequence of hand strokes,
with the (x, y) coordinates for each stroke path, we can use this information as
input to the classifier. This sequence of coordinates gives the steps that were
necessary to produce the final handwriting result, one tiny line at a time. It
gives more details about distinctive characteristics of the writer.

The most natural way to input this format is a sequence of images for each
sub-stroke, however, each handwriting has a few dozen strokes, with each stroke
having multiple sub-stroke lines. This makes is unpractical to use due to memory
issues and is probably not very efficient. This is why we will have to feed the
raw line coordinates as a sequence of (x, y) numbers to a sequential model like
a GRU.

![](https://cdn-images-1.medium.com/max/800/1*rOv_W4UqXqfk_WRXnahsQA.png)

<span class="figcaption_hack">Simplified illustration of the on-line model</span>


This has a much lighter memory usage and we will show that it can work well with
a much smaller model than the image encoder.

### Results :

* On-line Resnet Model Validation accuracy : **88%**
* Off-line GRU Model Validation accuracy : **92%**

This means that the GRU model can tell with 92% accuracy if two sets of
handwriting are made by the same writer or not when applied to unseen writers.
Those are encouraging results given the small size of the dataset.

### Possible Future Improvements :

There are multiple ways this project can be improved ->

* Parameter Tuning.
* Better (x, y) stroke sequence pre-processing and feature engineering.
* Looking for a more realistic dataset of handwriting ( ink pen on paper ) instead
of electronic recordings of handwritings.
* Using a dataset of handwriting forgery as hard negative examples for
training/evaluation.


### Conclusion :

This was a pretty fun application of machine leaning since it demonstrates that
each person’s handwriting may be identified. It also shows that there can be
multiple ways to model the same problem that can be considered before finding
the best approach.

Data :
[http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database/download-the-iam-on-line-handwriting-database](http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database/download-the-iam-on-line-handwriting-database)

Code :
[https://github.com/CVxTz/handwriting_forensics](https://github.com/CVxTz/handwriting_forensics)
