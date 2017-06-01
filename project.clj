(defproject image-nn "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [clojure-tensorflow "0.1.5"]
                 [net.mikera/imagez "0.12.0"]
                 ]
  ; :java-cmd "../../jdk1.8.0_05/bin/java" 
  ; :jvm-opts ["-Djava.library.path=../../tf-test/native"]
  :plugins [[cider/cider-nrepl "0.14.0"]]
  )
