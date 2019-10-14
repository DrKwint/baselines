;;;; dfa-compiler.lisp

(in-package #:dfa-compiler)

(defparameter +stages+ (list :re-to-nfa :nfa-to-dfa :minimize-dfa))

(defparameter *default-stages*
  (copy-list +stages+))

(defun start-dfa-compiler ()
  (dfa-compiler (argv0) (command-line-arguments)))

(defun parse-stages (string)
  (list))

(define-opts
  (:name :help
   :description "Show Help"
   :short #\h
   :long "help")
  (:name :repl
   :description "Open REPL with packages loaded."
   :short #\r
   :long "repl")
  (:name :swank
   :description "Start swank on PORT."
   :long "swank"
   :arg-parser #'parse-integer
   :meta-var "PORT")
  (:name :input-file
   :description "Input file name (regexp, in documented format)."
   :long "input-file"
   :short #\i
   :arg-parser #'identity
   :meta-var "INFILE")
  (:name :output-file
   :description "Output file name (DFA, minimized, documented format)."
   :long "output-file"
   :short #\o
   :arg-parser #'identity
   :meta-var "OUTFILE")
  (:name :stages
   :description "List of stages to run, comma separated."
   :long "stages"
   :arg-parser #'parse-stages
   :meta-var "STAGES"))

(defun dfa-compiler (program-name program-arguments)
  (multiple-value-bind (options free-arguments)
      (handler-case (get-opts program-arguments)
        (unknown-option (condition)
          (format *error-output*
                  "~&ERROR: Unknown option \"~A\".~%"
                  (option condition))
          (quit 1 t))
        (missing-arg (condition)
          (format *error-output*
                  "~&ERROR: Option \"~A\" is missing an argument.~%"
                  (option condition)) 
          (quit 1 t))
        (arg-parser-failed (condition)
          (format *error-output*
                  "~&ERROR: Option \"~A\" is unable to read its argument.~%"
                  (option condition))
          (quit 1 t))
        (missing-required-option (condition)
          (format *error-output*
                  "~&ERROR: Missing the following required options:~%~{~3T--~A~^~%~}~%"
                  (mapcar #'long (missing-options condition)))
          (quit 1 t)))
    
    (unless (null free-arguments)
      (format *error-output* "~&Too many additional arguments!~%")
      (quit 1 t))
    (when (getf options :repl)
      (format t "~&DFA Compile REPL.~%~%")
      (repl :nobanner t))
    (when (getf options :swank)
      (create-server :dont-close t
                     :port (getf options :swank))
      (repl :nobanner t))
    (when (getf options :help)
      (opts:describe :usage-of program-name)
      (quit 0 t))
    )
  (quit 0 t))
