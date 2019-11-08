;;;; dfa-compiler.asd

(asdf:defsystem #:dfa-compiler
  :description "RE to minimal DFA compiler"
  :author "Samuel W. Flint <swflint@flintfam.org>"
  :license  "Specify license here"
  :version "0.0.1"
  :serial t
  :depends-on (:uiop
               :unix-opts
               :prepl
               :swank)
  :components ((:file "package")
               (:file "regexp")
               (:file "nfa")
               (:file "dfa")
               (:file "thompsons-construction")
               (:file "dfa-compiler"))
  :build-operation "program-op"
  :build-pathname "dfa-compiler"
  :entry-point "dfa-compiler:start-dfa-compiler")

#+sb-core-compression
(defmethod asdf:perform ((o asdf:image-op) (c asdf:system))
  (uiop:dump-image (asdf:output-file o c) :executable t :compression t))
