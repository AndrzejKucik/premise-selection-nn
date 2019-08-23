#!/usr/bin/env python3.7

"""Tools for premise selection NN framework."""

# Built-in modules
import os
from datetime import timedelta
from random import sample
from time import time

# Third-party modules
import numpy as np
import pickle
from lark import Lark, Transformer
from tqdm import tqdm

# -- File info --
__version__ = '0.1.3'
__copyright__ = 'Andrzej Kucik 2019'
__author__ = 'Andrzej Kucik'
__maintainer__ = 'Andrzej Kucik'
__email__ = 'andrzej.kucik@gmail.com'
__date__ = '2019-08-21'

# TPTP parser
tptp_parser = Lark(r"""
    ?tptp_file : tptp_input*
    ?tptp_input : annotated_formula | include

    ?annotated_formula : thf_annotated | tfx_annotated | tff_annotated| tcf_annotated | fof_annotated | cnf_annotated
                       | tpi_annotated

    tpi_annotated : "tpi(" name "," formula_role "," tpi_formula annotations* ")."
    tpi_formula : fof_formula
    thf_annotated : "thf(" name "," formula_role "," thf_formula annotations* ")."
    tfx_annotated : "tfx("  name "," formula_role "," tfx_formula annotations* ")."
    tff_annotated : "tff(" name "," formula_role "," tff_formula annotations* ")."
    tcf_annotated : "tcf(" name "," formula_role "," tcf_formula annotations* ")."
    fof_annotated : "fof(" name "," formula_role "," fof_formula annotations* ")."
    cnf_annotated : "cnf(" name "," formula_role "," cnf_formula annotations* ")."
    annotations : "," source (optional_info)*

    formula_role : FORMULA_ROLE
    FORMULA_ROLE : "axiom" | "hypothesis" | "definition" | "assumption" | "lemma" | "theorem" | "corollary"
                 | "conjecture" | "negated_conjecture" | "plain" | "type" | "fi_domain" | "fi_functors"
                 | "fi_predicates" | "unknown" | LOWER_WORD

    thf_formula : thf_logic_formula | thf_sequent
    thf_logic_formula : thf_binary_formula | thf_unitary_formula | thf_type_formula | thf_subtype
    thf_binary_formula : thf_binary_pair | thf_binary_tuple | thf_binary_type

    thf_binary_pair : thf_unitary_formula thf_pair_connective thf_unitary_formula
    thf_binary_tuple : thf_or_formula | thf_and_formula | thf_apply_formula
    thf_or_formula : thf_unitary_formula "|" thf_unitary_formula | thf_or_formula "|" thf_unitary_formula
    thf_and_formula : thf_unitary_formula "&" thf_unitary_formula | thf_and_formula "&" thf_unitary_formula

    thf_apply_formula: thf_unitary_formula "@" thf_unitary_formula | thf_apply_formula "@" thf_unitary_formula

    thf_unitary_formula : thf_quantified_formula | thf_unary_formula | thf_atom | thf_conditional | thf_let | thf_tuple
                        | "(" thf_logic_formula ")"

    thf_quantified_formula : thf_quantification thf_unitary_formula
    thf_quantification : thf_quantifier "[" thf_variable_list "] :"
    thf_variable_list : thf_variable ("," thf_variable)*
    thf_variable : thf_typed_variable | variable
    thf_typed_variable : variable ":" thf_top_level_type

    thf_unary_formula : thf_unary_connective "(" thf_logic_formula ")"
    thf_atom : thf_function | variable | defined_term | thf_conn_term

    thf_function : atom | functor "(" thf_arguments ")" | defined_functor "(" thf_arguments ")"
                 | system_functor "(" thf_arguments ")"

    thf_conn_term : thf_pair_connective | assoc_connective | thf_unary_connective

    thf_conditional : "$ite(" thf_logic_formula "," thf_logic_formula "," thf_logic_formula ")"

    thf_let : "$let(" thf_unitary_formula "," thf_formula ")" | "$let(" thf_let_defns "," thf_formula")"
    thf_let_defns : thf_let_defn | "[" thf_let_defn_list "]"
    thf_let_defn_list : thf_let_defn ("," thf_let_defn)*
    thf_let_defn : thf_let_quantified_defn | thf_let_plain_defn
    thf_let_quantified_defn: thf_quantification "(" thf_let_plain_defn ")"
    thf_let_plain_defn: thf_let_defn_lhs ASSIGNMENT thf_formula
    thf_let_defn_lhs : constant | functor "(" fof_arguments ")" | thf_tuple

    thf_arguments : thf_formula_list

    thf_type_formula : thf_typeable_formula ":" thf_top_level_type | constant ":" thf_top_level_type
    thf_typeable_formula : thf_atom | "(" thf_logic_formula ")"
    thf_subtype : thf_atom "<<" thf_atom

    thf_top_level_type : thf_unitary_type | thf_mapping_type | thf_apply_type

    thf_unitary_type : thf_unitary_formula
    thf_apply_type : thf_apply_formula
    thf_binary_type : thf_mapping_type | thf_xprod_type | thf_union_type
    thf_mapping_type : thf_unitary_type ">" thf_unitary_type | thf_unitary_type ">" thf_mapping_type
    thf_xprod_type : thf_unitary_type "*" thf_unitary_type | thf_xprod_type "*" thf_unitary_type
    thf_union_type : thf_unitary_type "+" thf_unitary_type | thf_union_type "+" thf_unitary_type

    thf_sequent : thf_tuple "-->" thf_tuple | "(" thf_sequent ")"

    thf_tuple : "[" thf_formula_list? "]" | "{" thf_formula_list? "}"
    thf_formula_list : thf_logic_formula ("," thf_logic_formula)*

    logic_defn_rule : logic_defn_lhs assignment logic_defn_rhs
    logic_defn_lhs : logic_defn_value | thf_top_level_type | name | "$constants" | "$quantification" | "$consequence"
                   | "$modalities"

    logic_defn_rhs : logic_defn_value | thf_unitary_formula
    logic_defn_value : LOGIC_DEFN_VALUE
    LOGIC_DEFN_VALUE : DEFINED_CONSTANT | "$rigid" | "$flexible" | "$constant" | "$varying" | "$cumulative"
                     | "$decreasing" | "$local" | "$global" | "$modal_system_K" | "$modal_system_T" | "$modal_system_D"
                     | "$modal_system_S4" | "$modal_system_S5" | "$modal_axiom_K" | "$modal_axiom_T" | "$modal_axiom_B"
                     | "$modal_axiom_D" | "$modal_axiom_4" | "$modal_axiom_5"

    tfx_formula : tfx_logic_formula | thf_sequent
    tfx_logic_formula : thf_logic_formula

    tff_formula : tff_logic_formula | tff_typed_atom | tff_sequent
    tff_logic_formula : tff_binary_formula | tff_unitary_formula | tff_subtype
    tff_binary_formula : tff_binary_nonassoc | tff_binary_assoc
    tff_binary_nonassoc : tff_unitary_formula binary_connective tff_unitary_formula
    tff_binary_assoc : tff_or_formula | tff_and_formula
    tff_or_formula : tff_unitary_formula "|" tff_unitary_formula | tff_or_formula "|" tff_unitary_formula
    tff_and_formula : tff_unitary_formula "&" tff_unitary_formula | tff_and_formula "&" tff_unitary_formula
    tff_unitary_formula : tff_quantified_formula | tff_unary_formula | tff_atomic_formula | tff_conditional | tff_let
                        | "(" tff_logic_formula ")"

    tff_quantified_formula : fof_quantifier "[" tff_variable_list "] :" tff_unitary_formula
    tff_variable_list : tff_variable ("," tff_variable)*
    tff_variable : tff_typed_variable | variable
    tff_typed_variable : variable ":" tff_atomic_type
    tff_unary_formula : "~" tff_unitary_formula | fof_infix_unary
    tff_atomic_formula : fof_atomic_formula
    tff_conditional : "$ite_f(" tff_logic_formula "," tff_logic_formula "," tff_logic_formula ")"
    tff_let : "$let_tf(" tff_let_term_defns "," tff_formula ")" | "$let_ff(" tff_let_formula_defns "," tff_formula ")"

    tff_let_term_defns : tff_let_term_defn | "[" tff_let_term_list "]"
    tff_let_term_list : tff_let_term_defn ("," tff_let_term_defn)*
    tff_let_term_defn : "! [" tff_variable_list "] :" tff_let_term_defn | tff_let_term_binding
    tff_let_term_binding : fof_plain_term "=" fof_term | "(" tff_let_term_binding ")"
    tff_let_formula_defns : tff_let_formula_defn | "[" tff_let_formula_list "]"
    tff_let_formula_list : tff_let_formula_defn ("," tff_let_formula_defn)*
    tff_let_formula_defn : "! [ "tff_variable_list "] :" tff_let_formula_defn | tff_let_formula_binding
    tff_let_formula_binding : fof_plain_atomic_formula "<=>" tff_unitary_formula | "(" tff_let_formula_binding ")"
    tff_sequent : tff_formula_tuple "-->" tff_formula_tuple | "(" tff_sequent ")"
    tff_formula_tuple : "[" [tff_formula_tuple_list] "]"
    tff_formula_tuple_list : tff_logic_formula ("," tff_logic_formula)*

    tff_typed_atom : untyped_atom ":" tff_top_level_type | "(" tff_typed_atom ")"
    tff_subtype : untyped_atom "<<" atom

    tff_top_level_type : tff_atomic_type | tff_mapping_type | tf1_quantified_type | "(" tff_top_level_type ")"
    tf1_quantified_type : "!> [" tff_variable_list "] :" tff_monotype
    tff_monotype : tff_atomic_type | "(" tff_mapping_type ")"
    tff_unitary_type :  tff_atomic_type | "(" tff_xprod_type ")"
    tff_atomic_type : type_constant | defined_type | type_functor "(" tff_type_arguments ")" | variable
    tff_type_arguments : tff_atomic_type ("," tff_atomic_type)*

    tff_mapping_type : tff_unitary_type ">" tff_atomic_type
    tff_xprod_type : tff_unitary_type "*" tff_atomic_type | tff_xprod_type "*" tff_atomic_type


    tcf_formula : tcf_logic_formula | tff_typed_atom
    tcf_logic_formula : tcf_quantified_formula | cnf_formula
    tcf_quantified_formula : "! [" tff_variable_list "] :" cnf_formula


    ?fof_formula : fof_logic_formula | fof_sequent
    ?fof_logic_formula : fof_binary_formula | fof_unitary_formula

    ?fof_binary_formula :  fof_binary_nonassoc | fof_binary_assoc

    ?fof_binary_nonassoc : fof_unitary_formula binary_connective fof_unitary_formula

    ?fof_binary_assoc : fof_or_formula | fof_and_formula
    fof_or_formula : fof_unitary_formula "|" fof_unitary_formula | fof_or_formula "|" fof_unitary_formula
    fof_and_formula  : fof_unitary_formula "&" fof_unitary_formula | fof_and_formula "&"  fof_unitary_formula

    ?fof_unitary_formula : fof_quantified_formula | fof_unary_formula | fof_atomic_formula | "(" fof_logic_formula ")"

    ?fof_quantified_formula : fof_quantifier "[" fof_variable_list "] :" fof_unitary_formula
    ?fof_variable_list : variable ("," variable)*
    fof_unary_formula : "~" fof_unitary_formula | fof_infix_unary

    fof_infix_unary : fof_term infix_inequality fof_term
    ?fof_atomic_formula : fof_plain_atomic_formula | fof_defined_atomic_formula | fof_system_atomic_formula
    ?fof_plain_atomic_formula : fof_plain_term
    ?fof_defined_atomic_formula : fof_defined_plain_formula | fof_defined_infix_formula
    ?fof_defined_plain_formula : fof_defined_plain_term | defined_proposition | defined_predicate "(" fof_arguments ")"
    ?fof_defined_infix_formula : fof_term defined_infix_pred fof_term

    ?fof_system_atomic_formula : fof_system_term

    ?fof_plain_term : constant | functor "(" fof_arguments ")"

    ?fof_defined_term : defined_term | fof_defined_atomic_term
    ?fof_defined_atomic_term : fof_defined_plain_term

    ?fof_defined_plain_term : defined_constant | defined_functor "(" fof_arguments ")"

    ?fof_system_term : system_constant | system_functor "(" fof_arguments ")"

    ?fof_arguments : fof_term ("," fof_term)*

    ?fof_term : fof_function_term | variable | tff_conditional_term | tff_let_term | tff_tuple_term
    ?fof_function_term : fof_plain_term | fof_defined_term | fof_system_term

    tff_conditional_term : "$ite_t(" tff_logic_formula "," fof_term "," fof_term ")"
    tff_let_term : "let_ft(" tff_let_formula_defns "," fof_term ")" | "$let_tt(" tff_let_term_defns ","fof_term ")"
    tff_tuple_term : "{" [fof_arguments] "}"


    fof_sequent : fof_formula_tuple "-->" fof_formula_tuple | "(" fof_sequent ")"
    ?fof_formula_tuple : "[" [fof_formula_tuple_list] "]"
    ?fof_formula_tuple_list : fof_logic_formula ("," fof_logic_formula)*


    cnf_formula : disjunction | "(" disjunction ")"
    disjunction : literal ("|" literal)*
    literal : fof_atomic_formula | "~" fof_atomic_formula | fof_infix_unary

    thf_quantifier : fof_quantifier | th0_quantifier | th1_quantifier

    th1_quantifier : TH1_QUANTIFIER
    TH1_QUANTIFIER : "!>" | "?*"
    th0_quantifier : TH0_QUANTIFIER
    TH0_QUANTIFIER : "^" | "@+" | "@-"
    thf_pair_connective : infix_equality | infix_inequality | binary_connective | assignment
    thf_unary_connective : "~" | TH1_UNARY_CONNECTIVE
    TH1_UNARY_CONNECTIVE : "!!" | "??" | "@@+" | "@@-" | "@="

    fof_quantifier : FOF_QUANTIFIER
    FOF_QUANTIFIER : "!" | "?"
    binary_connective : BINARY_CONNECTIVE
    BINARY_CONNECTIVE : "<=>" | "=>" | "<=" | "<~>" | "~|" | "~&"
    assoc_connective : ASSOC_CONNECTIVE
    ASSOC_CONNECTIVE : "&" | "|"

    assignment : ASSIGNMENT
    ASSIGNMENT : ":="

    type_constant : TYPE_CONSTANT
    TYPE_CONSTANT : TYPE_FUNCTOR
    type_functor : TYPE_FUNCTOR
    TYPE_FUNCTOR : ATOMIC_WORD
    defined_type : DEFINED_TYPE
    DEFINED_TYPE : ATOMIC_DEFINED_WORD | "$oType" | "$o" | "$iType" | "$i" | "$tType" | "$real" | "$rat" | "$int"

    system_type : SYSTEM_TYPE
    SYSTEM_TYPE : ATOMIC_SYSTEM_WORD

    atom : ATOM
    ATOM : UNTYPED_ATOM | DEFINED_CONSTANT
    untyped_atom : UNTYPED_ATOM
    UNTYPED_ATOM : CONSTANT | SYSTEM_CONSTANT
    defined_proposition : DEFINED_PROPOSITION
    DEFINED_PROPOSITION :  ATOMIC_DEFINED_WORD | "$true" | "$false"
    defined_predicate : DEFINED_PREDICATE
    DEFINED_PREDICATE : ATOMIC_DEFINED_WORD | "$distinct" | "$less" | "$lesseq" | "$greater" | "$greatereq" | "$is_int"
                        | "$is_rat" | "$box_P" | "$box_i" | "$box_int" | "$box" | "$dia_P" | "$dia_i" | "$dia_int"
                        | "$dia"

    defined_infix_pred : infix_equality | assignment
    infix_equality : INFIX_EQUALITY
    INFIX_EQUALITY : "="
    infix_inequality : INFIX_INEQUALITY
    INFIX_INEQUALITY : "!="

    constant : CONSTANT
    CONSTANT : FUNCTOR
    functor : FUNCTOR
    FUNCTOR : ATOMIC_WORD
    system_constant : SYSTEM_CONSTANT
    SYSTEM_CONSTANT : SYSTEM_FUNCTOR
    system_functor : SYSTEM_FUNCTOR
    SYSTEM_FUNCTOR : ATOMIC_SYSTEM_WORD
    defined_constant : DEFINED_CONSTANT
    DEFINED_CONSTANT : DEFINED_FUNCTOR
    defined_functor : DEFINED_FUNCTOR
    DEFINED_FUNCTOR : ATOMIC_DEFINED_WORD |"$uminus" | "$sum" | "$difference" | "$product" | "$quotient" | "$quotient_e"
                    | "$quotient_t" | "$quotient_f" | "$remainder_e" | "$remainder_t" | "$remainder_f" | "$floor"
                    | "$ceiling" | "$truncate" | "$round" | "$to_int" | "$to_rat" | "$to_real"
    ?defined_term : number | DISTINCT_OBJECT
    variable : VARIABLE
    VARIABLE : UPPER_WORD

    source : general_term | dag_source | internal_source | external_source | "[" sources "]"
    sources : source ("," source)*
    dag_source : name | inference_record
    inference_record : "inference(" inference_rule "," useful_info "," inference_parents ")"
    inference_rule : INFERENCE_RULE
    INFERENCE_RULE : ATOMIC_WORD

    inference_parents : "[" parent_list* "]"
    parent_list : parent_info ("," parent_info)*
    parent_info : source parent_details*
    parent_details : general_list
    internal_source : "introduced(" intro_type optional_info* ")"
    intro_type : "definition" | "axiom_of_choice" | "tautology" | "assumption"

    external_source : file_source | theory | creator_source
    file_source : "file(" FILE_NAME FILE_INFO* ")"
    FILE_INFO : "," NAME
    theory : "theory(" THEORY_NAME optional_info* ")"
    THEORY_NAME : "equality" | "ac"

    creator_source : "creator(" CREATOR_NAME optional_info* ")"
    CREATOR_NAME : ATOMIC_WORD

    optional_info : "," useful_info
    useful_info : general_list | "[" info_items* "]"
    info_items : info_item ("," info_item)*
    info_item : formula_item | inference_item | general_function

    formula_item : DESCRIPTION_ITEM  | IQUOTE_ITEM
    DESCRIPTION_ITEM : "description(" ATOMIC_WORD ")"
    IQUOTE_ITEM : "iquote(" ATOMIC_WORD ")"

    inference_item : inference_status | assumptions_record | new_symbol_record | refutation
    inference_status : "status(" STATUS_VALUE ")" | inference_info

    STATUS_VALUE : "suc" | "unp" | "sap" | "esa" | "sat" | "fsa" | "thm" | "eqv" | "tac" | "wec" | "eth" | "tau"
                 | "wtc" | "wth" | "cax" | "sca" | "tca" | "wca" | "cup" | "csp" | "ecs" | "csa" | "cth" | "ceq"
                 | "unc" | "wcc" | "ect" | "fun" | "uns" | "wuc" | "wct" | "scc" | "uca" | "noc"

    inference_info : inference_rule "(" ATOMIC_WORD "," general_list ")"

    assumptions_record : "assumptions([" name_list "])"

    refutation : "refutation(" file_source ")"

    new_symbol_record : "new_symbols(" ATOMIC_WORD ", [" new_symbol_list "])"
    new_symbol_list : principal_symbol ("," principal_symbol)*

    principal_symbol : functor | variable


    include : "include(" FILE_NAME formula_selection* ")."
    formula_selection : ",[" name_list "]"
    name_list : name ("," name)*

    general_term : general_data | general_data ":" general_term | general_list
    general_data : ATOMIC_WORD | general_function | variable | number | DISTINCT_OBJECT | formula_data
                 | "bind(" variable "," formula_data ")"
    general_function : ATOMIC_WORD "(" general_terms ")"

    formula_data : "$thf(" thf_formula ")" | "$tff(" tff_formula ")" | "$fof(" fof_formula ")" | "$cnf(" cnf_formula ")"
                 | "$fot(" fof_term ")"
    general_list : "[" general_terms? "]"
    general_terms : general_term ("," general_term)*

    name : NAME
    NAME : ATOMIC_WORD | INTEGER

    ATOMIC_WORD : LOWER_WORD | SINGLE_QUOTED

    ATOMIC_DEFINED_WORD : "$" LOWER_WORD
    ATOMIC_SYSTEM_WORD : "$$" LOWER_WORD
    number : INTEGER | RATIONAL | REAL

    FILE_NAME : SINGLE_QUOTED

    COMMENT : COMMENT_LINE | COMMENT_BLOCK
    COMMENT_LINE : "%" PRINTABLE_CHAR*
    COMMENT_BLOCK : "/*" NOT_STAR_SLASH? "*"+ "/"
    NOT_STAR_SLASH : ("^*"* "*"+ "^/*") ("^*")*

    SINGLE_QUOTED : "'" SQ_CHAR+ "'"

    DISTINCT_OBJECT : "\"" DO_CHAR* "\""

    UPPER_WORD : UPPER_ALPHA ALPHA_NUMERIC*
    LOWER_WORD : LOWER_ALPHA ALPHA_NUMERIC*

    REAL : SIGN? DECIMAL_FRACTION | SIGN? DECIMAL_EXPONENT
    RATIONAL : SIGN? DECIMAL "/" POSITIVE_DECIMAL
    INTEGER : SIGN? DECIMAL
    DECIMAL : ZERO_NUMERIC | POSITIVE_DECIMAL
    POSITIVE_DECIMAL : NON_ZERO_NUMERIC NUMERIC*           
    DECIMAL_EXPONENT : DECIMAL EXPONENT EXP_INTEGER | DECIMAL_FRACTION EXPONENT EXP_INTEGER
    DECIMAL_FRACTION : DECIMAL DOT_DECIMAL
    DOT_DECIMAL : "." NUMERIC+
    EXP_INTEGER : SIGN? NUMERIC+

    DO_CHAR : (/["\40"-"\41", "\43"-"\133", "\135"-"\176"]/ | "\\\\ \" \\\\")


    SQ_CHAR : (/["\40"-"\46", "\50"-"\133", "\135"-"\176"]/ | "\\\\ ' \\\\")

    SIGN : "+" | "-"
    EXPONENT : "E" | "e"
    ZERO_NUMERIC : "0"
    NON_ZERO_NUMERIC : "1" .. "9"
    NUMERIC : "0" .. "9"
    LOWER_ALPHA : "a" .. "z"
    UPPER_ALPHA : "A" .. "Z"
    ALPHA_NUMERIC : LOWER_ALPHA | UPPER_ALPHA | NUMERIC | "_"
    PRINTABLE_CHAR : /["\32"-"\126"]/

    VIEWABLE_CHAR : "\n"

    %import common.WS  
    %ignore WS
    %ignore COMMENT

    """, start='tptp_file')

# TPTP transformer
class list_of_functions(Transformer):
    fof_annotated = lambda self, a: tuple([a[0], a[2]])
    fof_formula = lambda self, a: a
    fof_logic_formula = lambda self, a: a

    fof_binary_formula = lambda self, a: a

    fof_binary_nonassoc = lambda self, a: a

    fof_binary_assoc = lambda self, a: a
    fof_or_formula = lambda self, a: a
    fof_and_formula = lambda self, a: a

    fof_unitary_formula = lambda self, a: a

    fof_quantified_formula = lambda self, a: a
    fof_variable_list = lambda self, a: a
    fof_unary_formula = lambda self, a: a

    fof_infix_unary = lambda self, a: a
    fof_atomic_formula = lambda self, a: a
    fof_plain_atomic_formula = lambda self, a: a
    fof_defined_atomic_formula = lambda self, a: a
    fof_defined_plain_formula = lambda self, a: a
    fof_defined_infix_formula = lambda self, a: a

    fof_system_atomic_formula = lambda self, a: a

    fof_plain_term = lambda self, a: a

    fof_defined_term = lambda self, a: a
    fof_defined_atomic_term = lambda self, a: a

    fof_defined_plain_term = lambda self, a: a

    fof_system_term = lambda self, a: a

    fof_arguments = lambda self, a: a

    fof_term = lambda self, a: a
    fof_function_term = lambda self, a: a

    tff_conditional_term = lambda self, a: a
    tff_let_term = lambda self, a: a
    tff_tuple_term = lambda self, a: a

    fof_sequent = lambda self, a: a
    fof_formula_tuple = lambda self, a: a
    fof_formula_tuple_list = lambda self, a: a

    fof_quantifier = lambda self, a: []
    binary_connective = lambda self, a: []
    assoc_connective = lambda self, a: []

    assignment = lambda self, a: []

    type_constant = lambda self, a: a[0][:]
    type_functor = lambda self, a: a[0][:]
    defined_type = lambda self, a: a[0][:]

    system_type = lambda self, a: a[0][:]

    atom = lambda self, a: a
    untyped_atom = lambda self, a: a
    defined_proposition = lambda self, a: []
    defined_predicate = lambda self, a: a[0][:]

    defined_infix_pred = lambda self, a: []
    infix_equality = lambda self, a: []
    infix_inequality = lambda self, a: []

    constant = lambda self, a: a[0][:]
    functor = lambda self, a: a[0][:]
    system_constant = lambda self, a: a[0][:]
    system_functor = lambda self, a: a[0][:]
    defined_constant = lambda self, a: a[0][:]
    defined_functor = lambda self, a: a[0][:]
    defined_term = lambda self, a: a
    variable = lambda self, a: []

    name = lambda self, a: a[0][:]

    number = lambda self, a: []


def flatten_list(lst):
    """Function flattening list of lists.
       Arguments:
       lst       - list (of lists).

       Returns:
       flattened - flattened list."""

    flattened = []
    for l in lst:
        if isinstance(l, list):
            flattened += flatten_list(l)
        else:
            flattened.append(l)

    return flattened


def extract_functions(tptp_file):
    """Function extracting functional symbols from a TPTP FOF formula.
       Arguments:
       tptp_file         - TPTP file, that can be parsed by a TPTP parser.

       Returns:
       premise_name      - name of the TPTP FOF formula,
       premise_functions - list of functional symbols (expressed as strings) within the scope of the TPTP FOF formula.
    """

    # Parse TPTP file
    tree = tptp_parser.parse(tptp_file)

    # Transform it into a list of functions
    list_of_fun = list_of_functions().transform(tree)

    # The first element on the list is the name of the TPTP formula
    premise_name = list_of_fun[0]

    # The second element is a list (of lists) of functional symbols, which we flatten to be a list of strings.
    premise_functions = flatten_list(list_of_fun[1])

    return premise_name, premise_functions


def extract_premises(path_to_premises, save_dir):
    """Function converting nndata from https://github.com/JUrban/deepmath to dictionaries.
       Arguments:
       path_to_premises      - path to where nndata is saved,
       save_dir              - path to where to save the dictionaries.

       Returns:
       conjecture_signatures - dictionary with conjectures as keys and their functional signatures as values,
       axiom_signatures      - dictionary with axioms as keys and their functional signatures as values,
       useful_axioms         - dictionary with conjectures as keys and useful axioms as values,
       useless_axioms        - dictionary with conjectures as keys and useless axioms as values.
    """

    # Record time
    start_time = time()

    # Create placeholders
    conjecture_signatures = {}
    axiom_signatures = {}
    useful_axioms = {}
    useless_axioms = {}

    # Loop through premise files
    for file in tqdm(os.listdir(path_to_premises)):
        # Open file
        with open(os.path.join(path_to_premises, file), 'r') as current_file:
            problem = current_file.read()

        # Split file into premises
        premises = problem.split('\n')[:-1]

        # The first line is a conjecture
        conjecture = premises[0][2:]

        # Get conjecture name and its functional signature and save in a dictionary
        conjecture_name, conjecture_functions = extract_functions(conjecture)
        conjecture_signatures[conjecture_name] = conjecture_functions

        # Get conjecture's axioms
        useful_axioms[conjecture_name] = []
        useless_axioms[conjecture_name] = []

        for axiom in premises[1:]:
            # Get axiom name
            axiom_name = axiom[6:].split(', axiom, ')[0]

            # Check axiom's usefulness
            if axiom.startswith('+'):
                useful_axioms[conjecture_name].append(axiom_name)
            elif axiom.startswith('-'):
                useless_axioms[conjecture_name].append(axiom_name)

            # If axiom was not seen before, we need to add its signature to axiom_signatures
            if axiom_name not in axiom_signatures.keys():
                axiom_signatures[axiom_name] = extract_functions(axiom[2:])[1]

    # Rec ord end time
    end_time = time()

    # Save the data as dictionaries
    with open(os.path.join(save_dir, 'conjecture_signatures.pickle'), 'wb') as dictionary:
        pickle.dump(conjecture_signatures, dictionary, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dir, 'axiom_signatures.pickle'), 'wb') as dictionary:
        pickle.dump(axiom_signatures, dictionary, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dir, 'useful_axioms.pickle'), 'wb') as dictionary:
        pickle.dump(useful_axioms, dictionary, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dir, 'useless_axioms.pickle'), 'wb') as dictionary:
        pickle.dump(useless_axioms, dictionary, protocol=pickle.HIGHEST_PROTOCOL)

    # Print some statistics
    print('Formatting time:', str(timedelta(seconds=end_time - start_time)), 'hours.')
    print('Total number of conjectures:', len(conjecture_signatures))
    print('Total number of axioms:', len(axiom_signatures))


def get_all_used_functions(paths_to_signatures):
    """Function returning the list of all functional signatures (expressed as strings) within the scope of some FOF
       formulae.
       Arguments:
       paths_to_signatures - list of paths or a path to a signature file.

       Return:
       all_used_functions  - alphabetically sorted list of all functional signatures (strings) within files stored at
                             paths_to_signatures.
    """

    if not isinstance(paths_to_signatures, list):
        paths_to_signatures = [paths_to_signatures]

    # Functional symbols are collected in the form of a set, to avoid repetitions
    all_used_functions = set()

    for path in paths_to_signatures:
        with open(path, 'rb') as dictionary:
            signatures = pickle.load(dictionary)
            for signature in signatures.values():
                all_used_functions |= set(signature)

    print('Total number of used functional symbols:', len(all_used_functions))

    return sorted(list(all_used_functions))


def convert_to_integers(paths_to_signatures):
    """Function converting list of functions expressed as strings to a list of functions expressed as integers.
       Arguments:
       path_to_signatures - list of paths or a path to dictionary/-ies with FOF formulae names as keys and list of
                            functional symbols (expressed as strings) as values.
    """

    if not isinstance(paths_to_signatures, list):
        paths_to_signatures = [paths_to_signatures]

    # Get all used functions to know how many integers are needed to label the functional symbols.
    all_used_functions = get_all_used_functions(paths_to_signatures)

    for path in paths_to_signatures:
        new_signatures = {}
        with open(path, 'rb') as dictionary:
            signatures = pickle.load(dictionary)
        for key, value in tqdm(signatures.items()):
            new_signatures[key] = [all_used_functions.index(v) for v in value]

        new_path = '_int.'.join(path.split('.'))
        with open(new_path, 'wb') as dictionary:
            pickle.dump(new_signatures, dictionary, protocol=pickle.HIGHEST_PROTOCOL)


def count_functions(signature, functions):
    """Function counting occurrences of a given functional symbol within the scope of a functional signature.
       Arguments:
       signature - list of functional symbols (expressed as strings),
       functions - list of all available functional symbols (also as strings).
    """

    return [signature.count(functions[n]) for n in range(len(functions))]


def convert_to_count_signatures(paths_to_signatures):

    if not isinstance(paths_to_signatures, list):
        paths_to_signatures = [paths_to_signatures]

    all_used_functions = get_all_used_functions(paths_to_signatures)

    for path in paths_to_signatures:
        with open(path, 'rb') as dictionary:
            signatures = pickle.load(dictionary)
            signatures = dict([(key, count_functions(value, all_used_functions)) for key, value in signatures.items()])

        new_path = '_count.'.join(path.split('.'))
        with open(new_path, 'wb') as dictionary:
            pickle.dump(signatures, dictionary, protocol=pickle.HIGHEST_PROTOCOL)


def calculate_context_distribution(path_to_context):
    with open(path_to_context, 'rb') as dictionary:
        context = pickle.load(dictionary)

    context = np.array(list(context.values()), dtype='uint16')

    num_context, num_fun = context.shape

    print('Number of context premises: {}, number of functions: {}.'.format(num_context, num_fun))

    # Create network output placeholder
    output_data = np.zeros((num_fun, num_fun), dtype='float32')

    # Calculate probability distribution of functions which are in scope of the same premises
    for n in tqdm(range(num_fun)):
        output = np.array([context[m] for m in range(num_context) if context[m, n] != 0], dtype='float32')
        numerator = np.sum(output, axis=0)
        denominator = np.sum(numerator)
        if denominator == 0:
            output_data[n, n] = 1
        else:
            output_data[n] = numerator / denominator

    new_path = path_to_context.split('.')[0] + '_context_distribution.npy'
    np.save(new_path, output_data)


def embed_integers(path_to_signatures, weight):
    new_signatures = {}
    with open(path_to_signatures, 'rb') as dictionary:
        signatures = pickle.load(dictionary)
    for key, value in tqdm(signatures.items()):
        new_signatures[key] = np.array([weight[n] for n in value], dtype='float32')

    new_path = '_embed.'.join(path_to_signatures.split('.'))
    with open(new_path, 'wb') as dictionary:
        pickle.dump(new_signatures, dictionary, protocol=pickle.HIGHEST_PROTOCOL)


def embed_count(path_to_signatures, weight):
    with open(path_to_signatures, 'rb') as dictionary:
        signatures = pickle.load(dictionary)

    for key in tqdm(signatures.keys()):
        value = np.array(signatures[key], dtype='float32')
        signatures[key] = np.matmul(value, weight) / np.max(value)

    new_path = '_embed.'.join(path_to_signatures.split('.'))
    with open(new_path, 'wb') as dictionary:
        pickle.dump(signatures, dictionary, protocol=pickle.HIGHEST_PROTOCOL)


def form_train_sets(path_to_data, split=10, rnn=False, embedding_len=256, max_len=64, concat=True):
    if rnn:
        name = '_int_embed.pickle'
    else:
        name = '_count_embed.pickle'

    with open(os.path.join(path_to_data, 'conjecture_signatures' + name), 'rb') as dictionary:
        conjecture_signatures = pickle.load(dictionary)
    with open(os.path.join(path_to_data, 'axiom_signatures' + name), 'rb') as dictionary:
        axiom_signatures = pickle.load(dictionary)
    with open(os.path.join(path_to_data, 'useful_axioms.pickle'), 'rb') as dictionary:
        useful_axioms = pickle.load(dictionary)
    with open(os.path.join(path_to_data, 'useless_axioms.pickle'), 'rb') as dictionary:
        useless_axioms = pickle.load(dictionary)

    conjecture_names = sorted(list(conjecture_signatures.keys()))
    chunk_size = len(conjecture_names) / split

    for n in range(split):
        selected_conjectures = conjecture_names[int(n * chunk_size): int((n + 1) * chunk_size)]

        if rnn:
            x_chunk = np.zeros((1, 2, max_len, embedding_len), dtype='float32')
        else:
            x_chunk = np.zeros((1, 2, embedding_len), dtype='float32')

        y_chunk = np.zeros((1,), dtype='bool_')

        print('Chunk {}/{}'.format(n + 1, split))
        for conjecture_name in tqdm(selected_conjectures):
            conjecture = conjecture_signatures[conjecture_name]
            if rnn:
                conjecture = conjecture[:max_len]
                conjecture = np.pad(conjecture, ((0, max_len - conjecture.shape[0]), (0, 0)), mode='constant',
                                    constant_values=0)
            for axiom_name in useful_axioms[conjecture_name]:
                axiom = axiom_signatures[axiom_name]
                if rnn:
                    axiom = axiom[:max_len]
                    axiom = np.pad(axiom, ((0, max_len - axiom.shape[0]), (0, 0)), mode='constant', constant_values=0)
                x = np.expand_dims(np.stack([conjecture, axiom], axis=0), axis=0)
                x_chunk = np.concatenate([x_chunk, x], axis=0)
            for axiom_name in useless_axioms[conjecture_name]:
                axiom = axiom_signatures[axiom_name]
                if rnn:
                    axiom = axiom[:max_len]
                    axiom = np.pad(axiom, ((0, max_len - axiom.shape[0]), (0, 0)), mode='constant', constant_values=0)
                x = np.expand_dims(np.stack([conjecture, axiom], axis=0), axis=0)
                x_chunk = np.concatenate([x_chunk, x], axis=0)

            y_chunk = np.concatenate([y_chunk,
                                      np.ones((len(useful_axioms[conjecture_name]),), dtype='bool_'),
                                      np.zeros((len(useless_axioms[conjecture_name]),), dtype='bool_')])

        if rnn:
            name = '_rnn'
        else:
            name = ''

        if concat:
            try:
                x = np.load('data/x{}.npy'.format(name), mmap_mode='r')
                np.save('data/x{}.npy'.format(name), np.concatenate([x, x_chunk[1:]]))

                y = np.load('data/y{}.npy'.format(name), mmap_mode='r')
                np.save('data/y{}.npy'.format(name), np.concatenate([y, y_chunk[1:]]))

            except FileNotFoundError:
                np.save('data/x{}.npy'.format(name), x_chunk[1:])
                np.save('data/y{}.npy'.format(name), y_chunk[1:])

        else:
            np.save('data/x{}_{}.npy'.format(name, n), x_chunk[1:])
            np.save('data/y{}_{}.npy'.format(name, n), y_chunk[1:])


def get_test_indices(path_to_data, split=10):

    with open(os.path.join(path_to_data, 'conjecture_signatures.pickle'), 'rb') as dictionary:
        conjecture_signatures = pickle.load(dictionary)
    with open(os.path.join(path_to_data, 'useful_axioms.pickle'), 'rb') as dictionary:
        useful_axioms = pickle.load(dictionary)
    with open(os.path.join(path_to_data, 'useless_axioms.pickle'), 'rb') as dictionary:
        useless_axioms = pickle.load(dictionary)

    conjecture_names = sorted(list(conjecture_signatures.keys()))

    test_conjectures = sorted(sample(conjecture_names, len(conjecture_names)//split))

    test_indices = []
    for conjecture in test_conjectures:
        start = conjecture_names.index(conjecture)
        stop = start + 1
        stop += len(useful_axioms[conjecture])
        stop += len(useless_axioms[conjecture])
        test_indices += list(range(start, stop))

    return test_indices

