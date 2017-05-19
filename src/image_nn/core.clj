(ns image-nn.core
  (:require [mikera.image.core :as im]
            [mikera.image.colours :as colour]
            [clojure-tensorflow.core :refer [session-run]]
            [clojure-tensorflow.optimizers :as tf.opt]
            [clojure-tensorflow.save :as tf.save]
            [clojure-tensorflow.ops :as tf]
            [clojure-tensorflow.build :refer [default-graph]]
            ))

;; Training data
(def train-x (im/new-image 32 32))
(def make-random #(im/set-pixels train-x
               (int-array (repeatedly 1024 colour/rand-colour))))
(make-random)

(def prep-image
  (comp flatten (partial map colour/values-rgb) im/get-pixels))

(def train-y (im/resize
          (im/load-image-resource "trump.jpg") 32 32))




;; Model
(def in (tf/constant (map prep-image [train-x])))
(def rand-syn #(dec (* 2 (rand))))
(def syn-0 (tf/variable (repeatedly (* 3 1024) #(repeatedly 1024 rand-syn))
            {:node-name "Syn0"}))
(def syn-1 (tf/variable (repeatedly 1024 #(repeatedly 1024 rand-syn))
                        {:node-name "Syn1"}))
(def syn-2 (tf/variable (repeatedly 1024 #(repeatedly (* 3 1024) rand-syn))
                        {:node-name "Syn2"}))
(def hid-0 (tf/sigmoid (tf/matmul in syn-0)))
(def hid-1 (tf/sigmoid (tf/matmul hid-0 syn-1)))
(def out (tf/sigmoid (tf/matmul hid-1 syn-2)))

(def error (tf/pow (tf/sub (tf/constant [(prep-image train-y)]) out) (tf/constant 2.)))

(def optimizer (tf.opt/gradient-descent error syn-0 syn-1 syn-2))

(def sess (clojure-tensorflow.core/session))
(def sess-run (partial session-run sess))
(sess-run
 [;(tf/global-variables-initializer)
  (tf/mean (tf/mean error))])

;; train a bit
(dotimes [i 100]
  (sess-run
    [(repeat 5000 optimizer)
     (tf/mean (tf/mean error))])
  ;; save vars
  (tf.save/save-vars sess "resources/snapshot.clj" [out]))



;; reload trained vars from snapshot
(sess-run
 [(tf.save/load-vars "resources/snapshot.clj")
  (tf/mean (tf/mean error))])


;; view output
(def display (im/new-image 32 32))
;(im/set-pixels display
;               (int-array
;                (map (partial apply colour/rgb)
;                     (partition 3
;                                (first
;                                 (sess-run
;                                  [out]))))))

;(im/show display :zoom 10.0 :title "Trumpo")
