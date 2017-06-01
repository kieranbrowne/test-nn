(ns image-nn.core
  (:require [mikera.image.core :as im]
            [mikera.image.colours :as colour]
            [clojure-tensorflow.core :refer [session-run]]
            [clojure-tensorflow.optimizers :as tf.opt]
            [clojure-tensorflow.save :as tf.save]
            [clojure-tensorflow.ops :as tf]
            [clojure-tensorflow.build :refer [default-graph]]
            ))

(defn rand-img [w h]
  (repeatedly (* w h) #(flatten (colour/values-rgb (colour/rand-colour)))))

;; Training data
(def train-x (repeatedly 5 #(rand-img 32 32)))
(def train-y (im/resize
          (im/load-image-resource "trump.jpg") 32 32))

;; Model
(def in (tf/constant train-x))
(def rand-syn #(dec (* 2 (rand))))
(def syn-0 (tf/variable (repeatedly (* 3 1024) #(repeatedly 100 rand-syn))
            {:node-name "Syn0"}))
(def syn-1 (tf/variable (repeatedly 100 #(repeatedly 100 rand-syn))
                        {:node-name "Syn1"}))
(def syn-2 (tf/variable (repeatedly 100 #(repeatedly (* 3 1024) rand-syn))
                        {:node-name "Syn2"}))
(def hid-0 (tf/sigmoid (tf/matmul in syn-0)))
(def hid-1 (tf/sigmoid (tf/matmul hid-0 syn-1)))
(def out (tf/sigmoid (tf/matmul hid-1 syn-2)))

(def error (tf/pow (tf/sub (tf/constant [(prep-image train-y)]) out) (tf/constant 2.)))

(def optimizer (tf.opt/gradient-descent error syn-0 syn-1 syn-2))

(def sess (clojure-tensorflow.core/session))
(def sess-run (partial session-run sess))
(sess-run
 [(tf/global-variables-initializer)
  (tf/mean (tf/mean error))])

;; train a bit
(dotimes [i 1]
  (sess-run
    [(repeat 500 optimizer)
     (tf/mean (tf/mean error))])
  ;; save vars
  ;; (tf.save/save-vars sess "resources/snapshot.clj" [out])
  )



;; reload trained vars from snapshot
(sess-run
 [(tf.save/load-vars "resources/snapshot.clj")
  (tf/mean (tf/mean error))])


;; view output
(defn weights->RGB [weights]
  (int-array
   (map (partial apply colour/rgb)
        (partition 3 weights))))

(defn activations
  "Visualise activations as a grid of images"
  ([weights w h mode]
   (let [n (count (first weights))
         s (int (Math/ceil (Math/sqrt n)))
         out (im/new-image (* (inc w) s) (* (inc h) s))
         ws (map (partial partition (case mode :rgb 3 1)) (transpose weights))
         ]
     (doseq [row (range s) col (range s)]
       (doseq [x (range w) y (range h)]
         (when (< (+ (* col s) row) n)
           (im/set-pixel out
                         (+ (* row (inc w)) x)
                         (+ (* col (inc h)) y)
                         (apply (case mode :rgb colour/rgb #(colour/rgb % % %))
                                (nth
                                 (nth ws (+ (* col s) row))
                                 (+ (* y h) x))
                                ))))) out))
  ([weights w h] (activations weights w h :grayscale)))

(im/show
 (activations (sess-run [syn-0]) 32 32 :rgb))

(im/show
 (activations (sess-run [syn-1]) 10 10) :zoom 3.0)
(im/show
 (activations (sess-run [syn-2]) 10 10) :zoom 2.0)

(Math/sqrt)
(count
 (first
  (sess-run [syn-0])))

(first
 (map (partial partition 3)
      (transpose (sess-run [syn-0]))))


(defn transpose [m]
  (apply mapv vector m))

(count
 (transpose (sess-run [syn-0])))



(first
 (map count
         (sess-run
          [syn-0])))

 (.shape syn-0)


(colour/rgb 1 1 1)

(doseq [x (range 10) y (range 10)] (println x y))

(doseq [i (range 10)] (print i))


(def display (im/new-image 32 32))
(im/set-pixels
 display
 (weights->RGB
  (first
   (sess-run
    [out]))))


(im/show display :zoom 2.0 :title "Trumpo")


