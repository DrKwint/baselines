(defvar my-re (make-instance 'regexp:<regexp-concat>
                             :parts (list
                                     (make-instance 'regexp:<regexp-symbol>
                                                    :symbol 'a)
                                     (make-instance 'regexp:<regexp-zero-or-more>
                                                    :regexp
                                                    (make-instance 'regexp:<regexp-alternation>
                                                                   :options (list
                                                                             (make-instance 'regexp:<regexp-symbol>
                                                                                            :symbol 'a)
                                                                             (make-instance 'regexp:<regexp-symbol>
                                                                                            :symbol 'b))))
                                     (make-instance 'regexp:<regexp-symbol>
                                                    :symbol 'c))))

(defvar my-nfa (regexp-nfa:nfa-conversion my-re))
