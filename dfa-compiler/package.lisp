;;;; package.lisp

(defpackage #:dfa-compiler.utils
  (:use #:cl)
  (:nicknames #:utils)
  (:export #:ensure-list))

(defpackage #:dfa-compiler.representations.regexp
  (:use #:cl
        #:utils)
  (:nicknames #:regexp)
  (:export #:*regexp-as-latex*
           #:<regexp>
           #:<regexp-empty-string>
           #:<regexp-symbol>
           #:<regexp-concat>
           #:<regexp-alternation>
           #:<regexp-maybe>
           #:<regexp-zero-or-more>
           #:regexp-value))

(defpackage #:dfa-compiler.representations.nfa
  (:use #:cl
        #:utils
        #:regexp)
  (:import-from #:alexandria
                #:symbolicate)
  (:nicknames #:nfa)
  (:export #:<nfa-node>
           #:edges-out
           #:<nfa-edge>
           #:to-node
           #:from-node
           #:<nfa-re-edge>
           #:regexp
           #:<nfa-symbol-edge>
           #:<nfa-epsilon-edge>
           #:<nfa>
           #:start
           #:acceptp
           #:nodes
           #:edges
           #:make-re-nfa
           #:nfa-to-dot
           #:epsilon-closure))

(defpackage #:dfa-compiler.representations.dfa
  (:use #:cl
        #:utils
        #:nfa)
  (:import-from #:alexandria
                #:symbolicate)
  (:nicknames #:dfa)
  (:export #:<dfa-node>
           #:<dfa-multi-node>
           #:<dfa-edge>
           #:transitionable-symbols
           #:<dfa>))

(defpackage #:dfa-compiler.read-regexp
  (:use #:cl
        #:utils
        #:regexp)
  (:nicknames #:read-regexp)
  (:import-from #:esrap
                #:defrule
                #:parse)
  (:import-from #:parse-number
                #:parse-number))

(defpackage #:dfa-compiler.conversions.regexp-nfa
  (:use #:cl
        #:utils
        #:regexp
        #:nfa)
  (:nicknames #:regexp-nfa)
  (:export #:nfa-conversion))

(defpackage #:dfa-compiler.conversions.nfa-dfa
  (:use #:cl
        #:utils
        #:nfa
        #:dfa)
  (:nicknames #:nfa-dfa))

(defpackage #:dfa-compiler.dfa-minimizer
  (:use #:cl
        #:utils
        #:dfa))

(defpackage #:dfa-compiler
  (:use #:cl
        #:utils)
  (:import-from #:uiop
                #:quit
                #:argv0
                #:command-line-arguments)
  (:import-from #:unix-opts
                #:define-opts
                #:get-opts
                #:option
                #:missing-options
                #:long
                #:unknown-option
                #:missing-arg
                #:arg-parser-failed
                #:missing-required-option)
  (:import-from #:prepl
                #:repl)
  (:import-from #:swank
                #:create-server)
  (:export #:start-dfa-compiler))
