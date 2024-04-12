;;; Directory Local Variables            -*- no-byte-compile: t -*-
;;; For more information see (info "(emacs) Directory Variables")

((js-json-mode . ((eval . (progn
                            (adria-json-on-save-mode)))))
 (python-mode . ((eval . (progn
                           (add-hook 'before-save-hook #'adria-python-format-buffer nil t)))))
 (prog-mode . ((unison-profile . "lp-cleanba"))))
