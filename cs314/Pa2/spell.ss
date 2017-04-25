; *********************************************
; *  314 Principles of Programming Languages  *
; *  Spring 2017                              *
; *  Student Version                          *
; *********************************************

;; contains "ctv", "A", and "reduce" definitions
(load "include.ss")

;; contains simple dictionary definition
(load "dictionary.ss")

;; -----------------------------------------------------
;; HELPER FUNCTIONS

;; *** CODE FOR ANY HELPER FUNCTION GOES HERE ***


(define compute 
  (lambda (char val)
    (+ (ctv char) (* 29 val))
  )
)

;; -----------------------------------------------------
;; KEY FUNCTION
;;
;;

(define key
  (lambda (w)
    (reduce compute w 5187)
))

;; -----------------------------------------------------
;; EXAMPLE KEY VALUES
;  (key '(h e l l o))       = 106402241991
;;   (key '(m a y))           = 126526810
;;   (key '(t r e e f r o g)) = 2594908189083745

;; -----------------------------------------------------
;; HASH FUNCTION GENERATORS

;; value of parameter "size" should be a prime number
(define gen-hash-division-method
  (lambda (size) ;; range of values: 0..size-1
    (lambda (x) 
     (modulo (key x) size)
    )
  )
)

;; value of parameter "size" is not critical
;; Note: hash functions may return integer values in "real"
;;       format, e.g., 17.0 for 17

(define gen-hash-multiplication-method
  (lambda (size) ;; range of values: 0..size-1
    (lambda (x)
      (let ((k (key x)))
        (floor 
          (*
            size
            (- (* k A) (floor (* k A)))
          )
        )
      )
)))


;; -----------------------------------------------------
;; EXAMPLE HASH FUNCTIONS AND HASH FUNCTION LISTS

(define hash-1 (gen-hash-division-method 70111))
(define hash-2 (gen-hash-division-method 89997))
(define hash-3 (gen-hash-multiplication-method 7224))
(define hash-4 (gen-hash-multiplication-method 900))

(define hashfl-1 (list hash-1 hash-2 hash-3 hash-4))
(define hashfl-2 (list hash-1 hash-3))
(define hashfl-3 (list hash-2 hash-3))

;; -----------------------------------------------------
;; EXAMPLE HASH VALUES
;;   to test your hash function implementation
;;

;;(hash-1 '(h e l l o))       ==> 35616
;;  (hash-1 '(m a y))           ==> 46566
;;  (hash-1 '(t r e e f r o g)) ==> 48238
;;
;;  (hash-2 '(h e l l o))       ==> 48849
;;  (hash-2 '(m a y))           ==> 81025
;;  (hash-2 '(t r e e f r o g)) ==> 16708
;;
;;  (hash-3 '(h e l l o))       ==> 6331.0
;;  (hash-3 '(m a y))           ==> 2456.0
;;  (hash-3 '(t r e e f r o g)) ==> 1806.0
;;
;;  (hash-4 '(h e l l o))       ==> 788.0
;;  (hash-4 '(m a y))           ==> 306.0
;;  (hash-4 '(t r e e f r o g)) ==> 225.0


;; -----------------------------------------------------
;; SPELL CHECKER GENERATOR


;; generates bit vector
(define gen-vector
  (lambda (hashfunctionlist vec dict)
    (if (null? hashfunctionlist) vec
    (append vec (map (car hashfunctionlist) dict) (gen-vector (cdr hashfunctionlist) vec dict))
    )
  )
)

;; checks a word for all hash functions
(define check-word
  (lambda (word hashfunctionlist vec)
    (cond ((null? hashfunctionlist) #t)
          ((find-word ((car hashfunctionlist) word) vec) (check-word word (cdr hashfunctionlist) vec))
          (else #f)
    )
  )
)

;; Finds a word in the bit vector, given its key
(define find-word 
  (lambda (word vec)
    (cond ((null? vec) #f)
          ((= (floor word) (floor (car vec))) #t)
          (else (find-word word (cdr vec)))
    )
  )
)

;; returns a function which spellchecks for all hash functions for a given
;; dict
(define gen-checker
  (lambda (hashfunctionlist dict)
    (lambda (word) 
      (check-word 
        word 
        hashfunctionlist 
        (gen-vector hashfunctionlist '() dict)
        )
    )
  )
)


;; -----------------------------------------------------
;; EXAMPLE SPELL CHECKERS

(define checker-1 (gen-checker hashfl-1 dictionary))
(define checker-2 (gen-checker hashfl-2 dictionary))
(define checker-3 (gen-checker hashfl-3 dictionary))

;; EXAMPLE APPLICATIONS OF A SPELL CHECKER
;;
;; (checker-1 '(a b s o l v e d e d))
;; (checker-2 '(h e l l o))
;;  (checker-2 '(a r g g g g)) ==> #t  // false positive
