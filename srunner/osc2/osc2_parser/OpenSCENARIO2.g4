
grammar OpenSCENARIO2;

tokens { INDENT, DEDENT}


@lexer::header{
from antlr4.Token import CommonToken
import re
import importlib
# Allow languages to extend the lexer and parser, by loading the parser dynamically
module_path = __name__[:-5]
language_name = __name__.split('.')[-1]
language_name = language_name[:-5]  # Remove Lexer from name
LanguageParser = getattr(importlib.import_module('{}Parser'.format(module_path)), '{}Parser'.format(language_name))
}

@lexer::members {

@property
def tokens(self):
    try:
        return self._tokens
    except AttributeError:
        self._tokens = []
        return self._tokens

@property
def indents(self):
    try:
        return self._indents
    except AttributeError:
        self._indents = []
        return self._indents

@property
def opened(self):
    try:
        return self._opened
    except AttributeError:
        self._opened = 0
        return self._opened

@opened.setter
def opened(self, value):
    self._opened = value

@property
def lastToken(self):
    try:
        return self._lastToken
    except AttributeError:
        self._lastToken = None
        return self._lastToken

@lastToken.setter
def lastToken(self, value):
    self._lastToken = value

def reset(self):
    super().reset()
    self.tokens = []
    self.indents = []
    self.opened = 0
    self.lastToken = None

def emitToken(self, t):
    super().emitToken(t)
    self.tokens.append(t)

def nextToken(self):
    if self._input.LA(1) == Token.EOF and self.indents:
        for i in range(len(self.tokens)-1,-1,-1):
            if self.tokens[i].type == Token.EOF:
                self.tokens.pop(i)
        self.emitToken(self.commonToken(LanguageParser.NEWLINE, '\n'))
        while self.indents:
            self.emitToken(self.createDedent())
            self.indents.pop()
        self.emitToken(self.commonToken(LanguageParser.EOF, "<EOF>"))
    next = super().nextToken()
    if next.channel == Token.DEFAULT_CHANNEL:
        self.lastToken = next
    return next if not self.tokens else self.tokens.pop(0)

def createDedent(self):
    dedent = self.commonToken(LanguageParser.DEDENT, "")
    dedent.line = self.lastToken.line
    return dedent

def commonToken(self, type, text, indent=0):
    stop = self.getCharIndex()-1-indent
    start = (stop - len(text) + 1) if text else stop
    return CommonToken(self._tokenFactorySourcePair, type, super().DEFAULT_TOKEN_CHANNEL, start, stop)

@staticmethod
def getIndentationCount(spaces):
    count = 0
    for ch in spaces:
        if ch == '\t':
            count += 8 - (count % 8)
        else:
            count += 1
    return count

def atStartOfInput(self):
    return Lexer.column.fget(self) == 0 and Lexer.line.fget(self) == 1

}


//----------------------------------------
// Parser rules
/*Top-Level structure*/
osc_file
    :   preludeStatement* oscDeclaration* EOF
    ; 

preludeStatement
    :   importStatement              
    ;

importStatement
	: 'import' importReference NEWLINE
	|  NEWLINE;

importReference
	:   StringLiteral
	|   structuredIdentifier
	;

structuredIdentifier
	:   Identifier
	|   structuredIdentifier'.'Identifier
	;

oscDeclaration
	:   physicalTypeDeclaration 
	|   unitDeclaration
	|   enumDeclaration
	|   structDeclaration   
	|   actorDeclaration
	|   actionDeclaration
	|   scenarioDeclaration
	|   modifierDeclaration
	|   typeExtension
	|   globalParameterDeclaration
	|   NEWLINE
	;

//----------------------------------------
// physicalTypeDeclaration
physicalTypeDeclaration : 'type' physicalTypeName 'is' baseUnitSpecifier NEWLINE;

physicalTypeName : Identifier;

baseUnitSpecifier : sIBaseUnitSpecifier;

sIBaseUnitSpecifier : 'SI' OPEN_PAREN sIBaseExponentList CLOSE_PAREN;

//----------------------------------------
// unitDeclaration
unitDeclaration : 'unit' unitName= Identifier 'of' physicalTypeName 'is' unitSpecifier NEWLINE;

unitSpecifier : sIUnitSpecifier;

sIUnitSpecifier : 'SI' OPEN_PAREN sIBaseExponentList (',' sIFactor)? (',' sIOffset)? CLOSE_PAREN;

sIBaseExponentList : sIBaseExponent (',' sIBaseExponent)*;

//sIBaseUnitName : 'kg' | 'm' | 's' | 'A' | 'K' | 'mol' | 'cd' | 'rad';
sIBaseExponent : sIBaseUnitName = Identifier {
unitName = $sIBaseUnitName.text
if(not (unitName == "kg") and
	not (unitName == "m") and
	not (unitName == "s") and
	not (unitName == "A") and
	not (unitName == "K") and
	not (unitName == "mol") and
	not (unitName == "cd") and
	not (unitName == "factor") and
	not (unitName == "offset") and
	not (unitName == "rad")):
        print("unit name %s is not supported" %unitName)
        raise NoViableAltException(self)
	} ':' integerLiteral;

//sIFactor : 'factor' ':' (FloatLiteral | integerLiteral);
sIFactor : Identifier ':' (FloatLiteral | integerLiteral);

//sIOffset : 'offset' ':' (FloatLiteral | integerLiteral);
sIOffset : Identifier ':' (FloatLiteral | integerLiteral);

//----------------------------------------
// enumDeclaration
enumDeclaration : 'enum' enumName ':' OPEN_BRACK  enumMemberDecl (','  enumMemberDecl)* CLOSE_BRACK NEWLINE;

enumMemberDecl : enumMemberName ('=' enumMemberValue )?;

enumMemberValue: UintLiteral | HexUintLiteral;

enumName : Identifier;

enumMemberName : Identifier;

enumValueReference : enumName '!' enumMemberName;

//----------------------------------------
inheritsCondition : OPEN_PAREN fieldName '==' (enumValueReference | BoolLiteral) CLOSE_PAREN ;

//----------------------------------------
// structDeclaration
structDeclaration : 
	'struct' structName (structInherts)? 
	((':' NEWLINE INDENT structMemberDecl+ DEDENT) | NEWLINE);

structInherts : 'inherits' structName (inheritsCondition)? ;

structMemberDecl 
	:  eventDeclaration 
	|  fieldDeclaration 
	|  constraintDeclaration 
	|  methodDeclaration
	|  coverageDeclaration;

fieldName : Identifier;

structName : Identifier;

//----------------------------------------
// actorDeclaration
actorDeclaration :
	'actor' actorName (actorInherts)?
	((':' NEWLINE INDENT actorMemberDecl+ DEDENT) | NEWLINE);

actorInherts : 'inherits' actorName (inheritsCondition)? ;

actorMemberDecl
	:  eventDeclaration 
	|  fieldDeclaration 
	|  constraintDeclaration 
	|  methodDeclaration
	|  coverageDeclaration;

actorName : Identifier;

//----------------------------------------
// scenarioDeclaration
scenarioDeclaration 
	: 'scenario' qualifiedBehaviorName (scenarioInherts)?
	((':' NEWLINE INDENT
		(scenarioMemberDecl | behaviorSpecification )+
		DEDENT) | NEWLINE);

scenarioInherts : 'inherits' qualifiedBehaviorName (inheritsCondition)? ;

scenarioMemberDecl 
	: eventDeclaration 
	| fieldDeclaration 
	| constraintDeclaration 
	| methodDeclaration 
	| coverageDeclaration
	| modifierInvocation;

qualifiedBehaviorName : (actorName '.')? behaviorName;

behaviorName : Identifier;

//----------------------------------------
// actionDeclaration
actionDeclaration 
	:  'action' qualifiedBehaviorName (actionInherts)? 
	((':' NEWLINE INDENT (scenarioMemberDecl | behaviorSpecification)+ DEDENT) | NEWLINE);

actionInherts : 'inherits' qualifiedBehaviorName (inheritsCondition)? ;

//----------------------------------------
// modifierDeclaration
modifierDeclaration 
	: 'modifier' (actorName '.')? modifierName ('of'  qualifiedBehaviorName)?
	((':' NEWLINE INDENT scenarioMemberDecl+ DEDENT) | NEWLINE);

modifierName : Identifier;

//----------------------------------------
// typeExtension
typeExtension : enumTypeExtension | structuredTypeExtension;

/* 
enumTypeExtension : 'extend' enumName ':' NEWLINE INDENT
	(enumMemberDecl NEWLINE)+ DEDENT;
*/
enumTypeExtension : 'extend' enumName ':' OPEN_BRACK  enumMemberDecl (','  enumMemberDecl)* CLOSE_BRACK NEWLINE;


structuredTypeExtension : 'extend' extendableTypeName 
	':' NEWLINE INDENT extensionMemberDecl+ DEDENT;


extendableTypeName 
	: typeName 
	| qualifiedBehaviorName;

extensionMemberDecl 
	: structMemberDecl 
	| actorMemberDecl 
	| scenarioMemberDecl
	| behaviorSpecification;

//----------------------------------------
// globalParameterDeclaration
globalParameterDeclaration : 'global' fieldName (',' fieldName)* ':' typeDeclarator ('=' defaultValue)? (parameterWithDeclaration | NEWLINE);

//Type declarations
typeDeclarator : nonAggregateTypeDeclarator | aggregateTypeDeclarator;

nonAggregateTypeDeclarator : primitiveType | typeName | qualifiedBehaviorName;

aggregateTypeDeclarator : listTypeDeclarator;

listTypeDeclarator : 'list' 'of' nonAggregateTypeDeclarator;

primitiveType : 'int' | 'uint' | 'float' | 'bool' | 'string';

typeName : Identifier;

// Structured type members

// eventDeclaration
eventDeclaration 
	: 'event' eventName 
	(OPEN_PAREN argumentListSpecification CLOSE_PAREN)? 
	('is' eventSpecification)? NEWLINE;

eventSpecification 
	: eventReference (( eventFieldDecl )? 'if' eventCondition) ?
	| eventCondition ;

eventReference : '@' eventPath;
eventFieldDecl : 'as' eventFieldName;
eventFieldName : Identifier;
eventName : Identifier;
eventPath : (expression '.')? eventName;

eventCondition 
	:  boolExpression 
	|  riseExpression 
	|  fallExpression
	|  elapsedExpression 
	|  everyExpression;

riseExpression : 'rise' OPEN_PAREN boolExpression CLOSE_PAREN;
fallExpression :'fall' OPEN_PAREN boolExpression CLOSE_PAREN;
elapsedExpression : 'elapsed' OPEN_PAREN durationExpression CLOSE_PAREN;

//everyExpression : 'every' OPEN_PAREN durationExpression (',' 'offset' ':' durationExpression)? CLOSE_PAREN;
everyExpression : 'every' OPEN_PAREN durationExpression (',' Identifier{ 
offsetName = $Identifier.text
if( not (offsetName == "offset") ):
    print("%s must be offset" %offsetName)
    raise NoViableAltException(self)
} ':' durationExpression)? CLOSE_PAREN;

boolExpression : expression;
durationExpression : expression;

// fieldDeclaration
fieldDeclaration 
	:  parameterDeclaration 
	|  variableDeclaration;

//parameter-declaration ::= field-name (',' field-name)* ':' type-declarator ['=' default-value] [parameter-with-declaration] NEWLINE
//[improvement:] parameterWithDeclaration? NEWLINE -> (parameterWithDeclaration | NEWLINE)
parameterDeclaration 
	: fieldName (',' fieldName)* ':' typeDeclarator ('=' defaultValue)? (parameterWithDeclaration | NEWLINE); 

variableDeclaration 
	: 'var' fieldName (',' fieldName)* ':' typeDeclarator ('=' (sampleExpression | valueExp) )? NEWLINE;

sampleExpression 
	: 'sample' OPEN_PAREN expression ',' eventSpecification (',' defaultValue)? CLOSE_PAREN;

defaultValue : expression;

parameterWithDeclaration : 'with' ':' NEWLINE INDENT
	parameterWithMember+ DEDENT;

// add coverageDeclaration
parameterWithMember : constraintDeclaration | coverageDeclaration;

// constraintDeclaration
constraintDeclaration 
    : keepConstraintDeclaration | removeDefaultDeclaration;

keepConstraintDeclaration
    : 'keep' OPEN_PAREN (constraintQualifier)? constraintExpression CLOSE_PAREN NEWLINE;

constraintQualifier
    : 'default' | 'hard';

constraintExpression : expression;

removeDefaultDeclaration : 'remove_default' OPEN_PAREN parameterReference CLOSE_PAREN NEWLINE;

parameterReference : fieldName | fieldAccess ;

modifierInvocation : ((behaviorExpression | actorExpression) '.')? modifierName OPEN_PAREN (argumentList)? CLOSE_PAREN NEWLINE;

behaviorExpression : (actorExpression '.') behaviorName;

// behaviorSpecification
behaviorSpecification : onDirective | doDirective;

onDirective : 'on' eventSpecification ':' NEWLINE INDENT
	onMember+ DEDENT;

onMember : callDirective | emitDirective;

doDirective : 'do' doMember;

doMember 
	: (labelName ':')?(composition 
	| behaviorInvocation 
	| waitDirective 
	| emitDirective 
	| callDirective);

// composition
composition : compositionOperator (OPEN_PAREN argumentList? CLOSE_PAREN)?':' NEWLINE INDENT
	doMember+ DEDENT (behaviorWithDeclaration)?;

compositionOperator 
	: 'serial' | 'one_of' | 'parallel';

behaviorInvocation 
	: (actorExpression '.')? behaviorName OPEN_PAREN (argumentList)? CLOSE_PAREN (behaviorWithDeclaration | NEWLINE);

behaviorWithDeclaration : 'with' ':' NEWLINE INDENT
	behaviorWithMember+ DEDENT;

behaviorWithMember : constraintDeclaration
                   | modifierInvocation
                   | untilDirective;

labelName : Identifier;

actorExpression 
	: actorName;

waitDirective 
	: 'wait'  eventSpecification NEWLINE;

emitDirective :  'emit' eventName (OPEN_PAREN argumentList CLOSE_PAREN)? NEWLINE;

callDirective : 'call'  methodInvocation NEWLINE;

untilDirective : 'until' eventSpecification NEWLINE;

methodInvocation : postfixExp OPEN_PAREN (argumentList)? CLOSE_PAREN;

methodDeclaration : 'def' methodName OPEN_PAREN (argumentListSpecification)? CLOSE_PAREN ('->' returnType)? methodImplementation NEWLINE;

returnType : typeDeclarator;

methodImplementation 
	: 'is' (methodQualifier)? ('expression' expression 
	| 'undefined'
	| 'external' structuredIdentifier OPEN_PAREN (argumentList)? CLOSE_PAREN);

methodQualifier : 'only';

methodName : Identifier;

coverageDeclaration: coverDeclaration | recordDeclaration ;

coverDeclaration : 'cover' OPEN_PAREN targetName? coverageArgumentList* CLOSE_PAREN NEWLINE;

recordDeclaration : 'record' OPEN_PAREN targetName? coverageArgumentList* CLOSE_PAREN NEWLINE;

coverageArgumentList : (',' 'expression' ':' expression) #coverageExpression
                     | (',' 'unit' ':' unitName = Identifier) #coverageUnit
					 | (',' 'range' ':' rangeConstructor) #coverageRange
					 | (',' 'every' ':' valueExp) #coverageEvery
					 | (',' 'event' ':' eventName) #coverageEvent
					 | (',' namedArgument) #coverageNameArgument 
					 ;

targetName : Identifier ;

//Expressions
expression 
	: implication 
	| ternaryOpExp;

ternaryOpExp 
	: implication '?' expression ':' expression;

implication : disjunction ('=>' disjunction)*;
disjunction : conjunction ('or' conjunction)*;
conjunction : inversion ('and' inversion)*;
inversion 
	: 'not' inversion 
	| relation;

relation 
	: sum #sumExp
	| relation relationalOp sum #relationExp;

relationalOp : '==' | '!=' | '<' | '<=' | '>' | '>=' | 'in';

sum 
	: term #termExp
	| sum additiveOp term #additiveExp;

additiveOp 
	: '+' 
	| '-';

term 
	: factor #factorExp
	| term multiplicativeOp factor #multiplicativeExp;

multiplicativeOp 
	: '*' 
	| '/' 
	| '%';

factor 
	: postfixExp 
	| '-' factor;

postfixExp 
	: primaryExp #primaryExpression
	| postfixExp '.' 'as' OPEN_PAREN typeDeclarator CLOSE_PAREN #castExpression
	| postfixExp '.' 'is' OPEN_PAREN typeDeclarator CLOSE_PAREN #typeTestExpression
	| postfixExp OPEN_BRACK expression CLOSE_BRACK #elementAccessExpression
	| postfixExp OPEN_PAREN (argumentList)? CLOSE_PAREN #functionApplicationExpression
	| postfixExp '.' fieldName #fieldAccessExpression ; 

fieldAccess : postfixExp '.' fieldName ;
    
primaryExp 
	: valueExp
	| 'it' 
	| Identifier
	| OPEN_PAREN expression CLOSE_PAREN;

valueExp 
	: physicalLiteral
	| FloatLiteral
	| integerLiteral
	| BoolLiteral
	| StringLiteral
	| identifierReference
	| enumValueReference
	| listConstructor
	| rangeConstructor;

listConstructor : OPEN_BRACK expression (',' expression)* CLOSE_BRACK;
rangeConstructor 
	: 'range' OPEN_PAREN expression ',' expression CLOSE_PAREN 
	| OPEN_BRACK expression '..' expression CLOSE_BRACK;

identifierReference : (fieldName '.')* fieldName ;
//Common productions
argumentListSpecification : argumentSpecification (',' argumentSpecification)*;

argumentSpecification : argumentName ':' typeDeclarator ('=' defaultValue)?;

argumentName : Identifier;

argumentList 
	: positionalArgument (',' positionalArgument)* (',' namedArgument)*
	| namedArgument (',' namedArgument)*;

positionalArgument : expression;
namedArgument : argumentName ':' expression;

physicalLiteral : (FloatLiteral | integerLiteral) unitName= Identifier;

integerLiteral : UintLiteral | HexUintLiteral | IntLiteral;

//----------------------------------------
// Lexer rules

NEWLINE
 : ( {self.atStartOfInput()}?   SPACES
   | ( '\r'? '\n' | '\r' | '\f' ) SPACES?
   )
   {
tempt = Lexer.text.fget(self)
newLine = re.sub("[^\r\n\f]+", "", tempt)
spaces = re.sub("[\r\n\f]+", "", tempt)
la_char = ""
try:
    la = self._input.LA(1)
    la_char = chr(la)       # Python does not compare char to ints directly
except ValueError:          # End of file
    pass
# Strip newlines inside open clauses except if we are near EOF. We keep NEWLINEs near EOF to
# satisfy the final newline needed by the single_put rule used by the REPL.
try:
    nextnext_la = self._input.LA(2)
    nextnext_la_char = chr(nextnext_la)
except ValueError:
    nextnext_eof = True
else:
    nextnext_eof = False
if self.opened > 0 or nextnext_eof is False and (la_char == '\r' or la_char == '\n' or la_char == '\f' or la_char == '#'):
    self.skip()
else:
    indent = self.getIndentationCount(spaces)
    previous = self.indents[-1] if self.indents else 0
    self.emitToken(self.commonToken(self.NEWLINE, newLine, indent=indent))      # NEWLINE is actually the '\n' char
    if indent == previous:
        self.skip()
    elif indent > previous:
        self.indents.append(indent)
        self.emitToken(self.commonToken(LanguageParser.INDENT, spaces))
    else:
        while self.indents and self.indents[-1] > indent:
            self.emitToken(self.createDedent())
            self.indents.pop()
    }
 ;

OPEN_BRACK : '[' {self.opened += 1}  ;
CLOSE_BRACK : ']' {self.opened -= 1}  ;
OPEN_PAREN : '(' {self.opened += 1}  ;
CLOSE_PAREN : ')' {self.opened -= 1}  ;
 

SKIP_
 : (SPACES | LINE_JOINING)
 ->skip
 ;

 fragment 
 SPACES
 : [ \t]+
 ;
 
fragment LINE_JOINING
 : '\\' SPACES? '\r'? '\n'
 ;

fragment 
RN
    : '\r'? '\n'
    ;

BLOCK_COMMENT
    :   '/*' .*? '*/'
        -> skip
    ;

LINE_COMMENT
    :   '#' ~[\r\n\f]*
        -> skip
    ;

StringLiteral
	:   Shortstring 
	|   Longstring
	;

fragment
Shortstring
	: ('"' ShortstringElem* '"') | ('\'' ShortstringElem* '\'');


fragment
ShortstringElem 
	:	ShortstringChar | StringEscapeSeq;

fragment
ShortstringChar
	: ~[\\'"\r\n];


fragment
Longstring 
    :   ('"""' LongstringElem* '"""') | ('\'\'\'' LongstringElem* '\'\'\'');

fragment
LongstringElem : LongstringChar | StringEscapeSeq;

fragment
LongstringChar : ~'\\';

fragment
StringEscapeSeq
	: '\\'.
	| '\\' RN // [improvement:] consider: '\r'? '\n'
	;


FloatLiteral : [+-]? Digit* '.' Digit+ ([eE] [+-]? Digit+)?;

UintLiteral : Digit+;

HexUintLiteral : '0x' HexDigit+;

IntLiteral : '-' Digit+;

BoolLiteral :
  'true' | 'false';


/*
where `id-start-char` matches all characters of the following Unicode character categories:

* Ll -- Lowercase Letter
* Lm -- Modifier Letter
* Lo -- Other Letter
* Lt -- Titlecase Letter
* Lu -- Uppercase Letter
* Nl -- Letter Number

It also matches the underscore character `_` (U+005F).

`id-char` matches all characters that `id-start-char` matches, and additionally all characters of the following Unicode character categories:

* Mc -- Spacing Combining Mark
* Mn -- Nonspacing Mark
* Nd -- Decimal Number
* Pc -- Connector Punctuations

`non-vertical-line-char` matches all Unicode characters, except the vertical line `|` (U+007C) character.
*/
Identifier
	:	IdStartChar (IdChar)*
	|   '|' NonVerticalLineChar+ '|'
	;
	
fragment
NonVerticalLineChar : ~[\u007C];

fragment
Digit
	: [0-9]
	;

fragment
HexDigit
	: [0-9A-Fa-f]
	;

/// All characters in general categories Lu, Ll, Lt, Lm, Lo, Nl, the underscore, and [improvement:]characters with the Other_ID_Start property
fragment IdStartChar:
	'_'
	| [A-Z]
	| [a-z]
	| '\u00AA'
	| '\u00B5'
	| '\u00BA'
	| [\u00C0-\u00D6]
	| [\u00D8-\u00F6]
	| [\u00F8-\u01BA]
	| '\u01BB'
	| [\u01BC-\u01BF]
	| [\u01C0-\u01C3]
	| [\u01C4-\u0241]
	| [\u0250-\u02AF]
	| [\u02B0-\u02C1]
	| [\u02C6-\u02D1]
	| [\u02E0-\u02E4]
	| '\u02EE'
	| '\u037A'
	| '\u0386'
	| [\u0388-\u038A]
	| '\u038C'
	| [\u038E-\u03A1]
	| [\u03A3-\u03CE]
	| [\u03D0-\u03F5]
	| [\u03F7-\u0481]
	| [\u048A-\u04CE]
	| [\u04D0-\u04F9]
	| [\u0500-\u050F]
	| [\u0531-\u0556]
	| '\u0559'
	| [\u0561-\u0587]
	| [\u05D0-\u05EA]
	| [\u05F0-\u05F2]
	| [\u0621-\u063A]
	| '\u0640'
	| [\u0641-\u064A]
	| [\u066E-\u066F]
	| [\u0671-\u06D3]
	| '\u06D5'
	| [\u06E5-\u06E6]
	| [\u06EE-\u06EF]
	| [\u06FA-\u06FC]
	| '\u06FF'
	| '\u0710'
	| [\u0712-\u072F]
	| [\u074D-\u076D]
	| [\u0780-\u07A5]
	| '\u07B1'
	| [\u0904-\u0939]
	| '\u093D'
	| '\u0950'
	| [\u0958-\u0961]
	| '\u097D'
	| [\u0985-\u098C]
	| [\u098F-\u0990]
	| [\u0993-\u09A8]
	| [\u09AA-\u09B0]
	| '\u09B2'
	| [\u09B6-\u09B9]
	| '\u09BD'
	| '\u09CE'
	| [\u09DC-\u09DD]
	| [\u09DF-\u09E1]
	| [\u09F0-\u09F1]
	| [\u0A05-\u0A0A]
	| [\u0A0F-\u0A10]
	| [\u0A13-\u0A28]
	| [\u0A2A-\u0A30]
	| [\u0A32-\u0A33]
	| [\u0A35-\u0A36]
	| [\u0A38-\u0A39]
	| [\u0A59-\u0A5C]
	| '\u0A5E'
	| [\u0A72-\u0A74]
	| [\u0A85-\u0A8D]
	| [\u0A8F-\u0A91]
	| [\u0A93-\u0AA8]
	| [\u0AAA-\u0AB0]
	| [\u0AB2-\u0AB3]
	| [\u0AB5-\u0AB9]
	| '\u0ABD'
	| '\u0AD0'
	| [\u0AE0-\u0AE1]
	| [\u0B05-\u0B0C]
	| [\u0B0F-\u0B10]
	| [\u0B13-\u0B28]
	| [\u0B2A-\u0B30]
	| [\u0B32-\u0B33]
	| [\u0B35-\u0B39]
	| '\u0B3D'
	| [\u0B5C-\u0B5D]
	| [\u0B5F-\u0B61]
	| '\u0B71'
	| '\u0B83'
	| [\u0B85-\u0B8A]
	| [\u0B8E-\u0B90]
	| [\u0B92-\u0B95]
	| [\u0B99-\u0B9A]
	| '\u0B9C'
	| [\u0B9E-\u0B9F]
	| [\u0BA3-\u0BA4]
	| [\u0BA8-\u0BAA]
	| [\u0BAE-\u0BB9]
	| [\u0C05-\u0C0C]
	| [\u0C0E-\u0C10]
	| [\u0C12-\u0C28]
	| [\u0C2A-\u0C33]
	| [\u0C35-\u0C39]
	| [\u0C60-\u0C61]
	| [\u0C85-\u0C8C]
	| [\u0C8E-\u0C90]
	| [\u0C92-\u0CA8]
	| [\u0CAA-\u0CB3]
	| [\u0CB5-\u0CB9]
	| '\u0CBD'
	| '\u0CDE'
	| [\u0CE0-\u0CE1]
	| [\u0D05-\u0D0C]
	| [\u0D0E-\u0D10]
	| [\u0D12-\u0D28]
	| [\u0D2A-\u0D39]
	| [\u0D60-\u0D61]
	| [\u0D85-\u0D96]
	| [\u0D9A-\u0DB1]
	| [\u0DB3-\u0DBB]
	| '\u0DBD'
	| [\u0DC0-\u0DC6]
	| [\u0E01-\u0E30]
	| [\u0E32-\u0E33]
	| [\u0E40-\u0E45]
	| '\u0E46'
	| [\u0E81-\u0E82]
	| '\u0E84'
	| [\u0E87-\u0E88]
	| '\u0E8A'
	| '\u0E8D'
	| [\u0E94-\u0E97]
	| [\u0E99-\u0E9F]
	| [\u0EA1-\u0EA3]
	| '\u0EA5'
	| '\u0EA7'
	| [\u0EAA-\u0EAB]
	| [\u0EAD-\u0EB0]
	| [\u0EB2-\u0EB3]
	| '\u0EBD'
	| [\u0EC0-\u0EC4]
	| '\u0EC6'
	| [\u0EDC-\u0EDD]
	| '\u0F00'
	| [\u0F40-\u0F47]
	| [\u0F49-\u0F6A]
	| [\u0F88-\u0F8B]
	| [\u1000-\u1021]
	| [\u1023-\u1027]
	| [\u1029-\u102A]
	| [\u1050-\u1055]
	| [\u10A0-\u10C5]
	| [\u10D0-\u10FA]
	| '\u10FC'
	| [\u1100-\u1159]
	| [\u115F-\u11A2]
	| [\u11A8-\u11F9]
	| [\u1200-\u1248]
	| [\u124A-\u124D]
	| [\u1250-\u1256]
	| '\u1258'
	| [\u125A-\u125D]
	| [\u1260-\u1288]
	| [\u128A-\u128D]
	| [\u1290-\u12B0]
	| [\u12B2-\u12B5]
	| [\u12B8-\u12BE]
	| '\u12C0'
	| [\u12C2-\u12C5]
	| [\u12C8-\u12D6]
	| [\u12D8-\u1310]
	| [\u1312-\u1315]
	| [\u1318-\u135A]
	| [\u1380-\u138F]
	| [\u13A0-\u13F4]
	| [\u1401-\u166C]
	| [\u166F-\u1676]
	| [\u1681-\u169A]
	| [\u16A0-\u16EA]
	| [\u16EE-\u16F0]
	| [\u1700-\u170C]
	| [\u170E-\u1711]
	| [\u1720-\u1731]
	| [\u1740-\u1751]
	| [\u1760-\u176C]
	| [\u176E-\u1770]
	| [\u1780-\u17B3]
	| '\u17D7'
	| '\u17DC'
	| [\u1820-\u1842]
	| '\u1843'
	| [\u1844-\u1877]
	| [\u1880-\u18A8]
	| [\u1900-\u191C]
	| [\u1950-\u196D]
	| [\u1970-\u1974]
	| [\u1980-\u19A9]
	| [\u19C1-\u19C7]
	| [\u1A00-\u1A16]
	| [\u1D00-\u1D2B]
	| [\u1D2C-\u1D61]
	| [\u1D62-\u1D77]
	| '\u1D78'
	| [\u1D79-\u1D9A]
	| [\u1D9B-\u1DBF]
	| [\u1E00-\u1E9B]
	| [\u1EA0-\u1EF9]
	| [\u1F00-\u1F15]
	| [\u1F18-\u1F1D]
	| [\u1F20-\u1F45]
	| [\u1F48-\u1F4D]
	| [\u1F50-\u1F57]
	| '\u1F59'
	| '\u1F5B'
	| '\u1F5D'
	| [\u1F5F-\u1F7D]
	| [\u1F80-\u1FB4]
	| [\u1FB6-\u1FBC]
	| '\u1FBE'
	| [\u1FC2-\u1FC4]
	| [\u1FC6-\u1FCC]
	| [\u1FD0-\u1FD3]
	| [\u1FD6-\u1FDB]
	| [\u1FE0-\u1FEC]
	| [\u1FF2-\u1FF4]
	| [\u1FF6-\u1FFC]
	| '\u2071'
	| '\u207F'
	| [\u2090-\u2094]
	| '\u2102'
	| '\u2107'
	| [\u210A-\u2113]
	| '\u2115'
	| '\u2118'
	| [\u2119-\u211D]
	| '\u2124'
	| '\u2126'
	| '\u2128'
	| [\u212A-\u212D]
	| '\u212E'
	| [\u212F-\u2131]
	| [\u2133-\u2134]
	| [\u2135-\u2138]
	| '\u2139'
	| [\u213C-\u213F]
	| [\u2145-\u2149]
	| [\u2160-\u2183]
	| [\u2C00-\u2C2E]
	| [\u2C30-\u2C5E]
	| [\u2C80-\u2CE4]
	| [\u2D00-\u2D25]
	| [\u2D30-\u2D65]
	| '\u2D6F'
	| [\u2D80-\u2D96]
	| [\u2DA0-\u2DA6]
	| [\u2DA8-\u2DAE]
	| [\u2DB0-\u2DB6]
	| [\u2DB8-\u2DBE]
	| [\u2DC0-\u2DC6]
	| [\u2DC8-\u2DCE]
	| [\u2DD0-\u2DD6]
	| [\u2DD8-\u2DDE]
	| '\u3005'
	| '\u3006'
	| '\u3007'
	| [\u3021-\u3029]
	| [\u3031-\u3035]
	| [\u3038-\u303A]
	| '\u303B'
	| '\u303C'
	| [\u3041-\u3096]
	| [\u309B-\u309C]
	| [\u309D-\u309E]
	| '\u309F'
	| [\u30A1-\u30FA]
	| [\u30FC-\u30FE]
	| '\u30FF'
	| [\u3105-\u312C]
	| [\u3131-\u318E]
	| [\u31A0-\u31B7]
	| [\u31F0-\u31FF]
	| [\u3400-\u4DB5]
	| [\u4E00-\u9FBB]
	| [\uA000-\uA014]
	| '\uA015'
	| [\uA016-\uA48C]
	| [\uA800-\uA801]
	| [\uA803-\uA805]
	| [\uA807-\uA80A]
	| [\uA80C-\uA822]
	| [\uAC00-\uD7A3]
	| [\uF900-\uFA2D]
	| [\uFA30-\uFA6A]
	| [\uFA70-\uFAD9]
	| [\uFB00-\uFB06]
	| [\uFB13-\uFB17]
	| '\uFB1D'
	| [\uFB1F-\uFB28]
	| [\uFB2A-\uFB36]
	| [\uFB38-\uFB3C]
	| '\uFB3E'
	| [\uFB40-\uFB41]
	| [\uFB43-\uFB44]
	| [\uFB46-\uFBB1]
	| [\uFBD3-\uFD3D]
	| [\uFD50-\uFD8F]
	| [\uFD92-\uFDC7]
	| [\uFDF0-\uFDFB]
	| [\uFE70-\uFE74]
	| [\uFE76-\uFEFC]
	| [\uFF21-\uFF3A]
	| [\uFF41-\uFF5A]
	| [\uFF66-\uFF6F]
	| '\uFF70'
	| [\uFF71-\uFF9D]
	| [\uFF9E-\uFF9F]
	| [\uFFA0-\uFFBE]
	| [\uFFC2-\uFFC7]
	| [\uFFCA-\uFFCF]
	| [\uFFD2-\uFFD7]
	| [\uFFDA-\uFFDC];

/// id_continue  ::=  <all characters in id_start, plus characters in the categories Mn, Mc, Nd, Pc and [improvement:] others with the Other_ID_Continue property>
fragment IdChar:
	IdStartChar
	| [0-9]
	| [\u0300-\u036F]
	| [\u0483-\u0486]
	| [\u0591-\u05B9]
	| [\u05BB-\u05BD]
	| '\u05BF'
	| [\u05C1-\u05C2]
	| [\u05C4-\u05C5]
	| '\u05C7'
	| [\u0610-\u0615]
	| [\u064B-\u065E]
	| [\u0660-\u0669]
	| '\u0670'
	| [\u06D6-\u06DC]
	| [\u06DF-\u06E4]
	| [\u06E7-\u06E8]
	| [\u06EA-\u06ED]
	| [\u06F0-\u06F9]
	| '\u0711'
	| [\u0730-\u074A]
	| [\u07A6-\u07B0]
	| [\u0901-\u0902]
	| '\u0903'
	| '\u093C'
	| [\u093E-\u0940]
	| [\u0941-\u0948]
	| [\u0949-\u094C]
	| '\u094D'
	| [\u0951-\u0954]
	| [\u0962-\u0963]
	| [\u0966-\u096F]
	| '\u0981'
	| [\u0982-\u0983]
	| '\u09BC'
	| [\u09BE-\u09C0]
	| [\u09C1-\u09C4]
	| [\u09C7-\u09C8]
	| [\u09CB-\u09CC]
	| '\u09CD'
	| '\u09D7'
	| [\u09E2-\u09E3]
	| [\u09E6-\u09EF]
	| [\u0A01-\u0A02]
	| '\u0A03'
	| '\u0A3C'
	| [\u0A3E-\u0A40]
	| [\u0A41-\u0A42]
	| [\u0A47-\u0A48]
	| [\u0A4B-\u0A4D]
	| [\u0A66-\u0A6F]
	| [\u0A70-\u0A71]
	| [\u0A81-\u0A82]
	| '\u0A83'
	| '\u0ABC'
	| [\u0ABE-\u0AC0]
	| [\u0AC1-\u0AC5]
	| [\u0AC7-\u0AC8]
	| '\u0AC9'
	| [\u0ACB-\u0ACC]
	| '\u0ACD'
	| [\u0AE2-\u0AE3]
	| [\u0AE6-\u0AEF]
	| '\u0B01'
	| [\u0B02-\u0B03]
	| '\u0B3C'
	| '\u0B3E'
	| '\u0B3F'
	| '\u0B40'
	| [\u0B41-\u0B43]
	| [\u0B47-\u0B48]
	| [\u0B4B-\u0B4C]
	| '\u0B4D'
	| '\u0B56'
	| '\u0B57'
	| [\u0B66-\u0B6F]
	| '\u0B82'
	| [\u0BBE-\u0BBF]
	| '\u0BC0'
	| [\u0BC1-\u0BC2]
	| [\u0BC6-\u0BC8]
	| [\u0BCA-\u0BCC]
	| '\u0BCD'
	| '\u0BD7'
	| [\u0BE6-\u0BEF]
	| [\u0C01-\u0C03]
	| [\u0C3E-\u0C40]
	| [\u0C41-\u0C44]
	| [\u0C46-\u0C48]
	| [\u0C4A-\u0C4D]
	| [\u0C55-\u0C56]
	| [\u0C66-\u0C6F]
	| [\u0C82-\u0C83]
	| '\u0CBC'
	| '\u0CBE'
	| '\u0CBF'
	| [\u0CC0-\u0CC4]
	| '\u0CC6'
	| [\u0CC7-\u0CC8]
	| [\u0CCA-\u0CCB]
	| [\u0CCC-\u0CCD]
	| [\u0CD5-\u0CD6]
	| [\u0CE6-\u0CEF]
	| [\u0D02-\u0D03]
	| [\u0D3E-\u0D40]
	| [\u0D41-\u0D43]
	| [\u0D46-\u0D48]
	| [\u0D4A-\u0D4C]
	| '\u0D4D'
	| '\u0D57'
	| [\u0D66-\u0D6F]
	| [\u0D82-\u0D83]
	| '\u0DCA'
	| [\u0DCF-\u0DD1]
	| [\u0DD2-\u0DD4]
	| '\u0DD6'
	| [\u0DD8-\u0DDF]
	| [\u0DF2-\u0DF3]
	| '\u0E31'
	| [\u0E34-\u0E3A]
	| [\u0E47-\u0E4E]
	| [\u0E50-\u0E59]
	| '\u0EB1'
	| [\u0EB4-\u0EB9]
	| [\u0EBB-\u0EBC]
	| [\u0EC8-\u0ECD]
	| [\u0ED0-\u0ED9]
	| [\u0F18-\u0F19]
	| [\u0F20-\u0F29]
	| '\u0F35'
	| '\u0F37'
	| '\u0F39'
	| [\u0F3E-\u0F3F]
	| [\u0F71-\u0F7E]
	| '\u0F7F'
	| [\u0F80-\u0F84]
	| [\u0F86-\u0F87]
	| [\u0F90-\u0F97]
	| [\u0F99-\u0FBC]
	| '\u0FC6'
	| '\u102C'
	| [\u102D-\u1030]
	| '\u1031'
	| '\u1032'
	| [\u1036-\u1037]
	| '\u1038'
	| '\u1039'
	| [\u1040-\u1049]
	| [\u1056-\u1057]
	| [\u1058-\u1059]
	| '\u135F'
	| [\u1369-\u1371]
	| [\u1712-\u1714]
	| [\u1732-\u1734]
	| [\u1752-\u1753]
	| [\u1772-\u1773]
	| '\u17B6'
	| [\u17B7-\u17BD]
	| [\u17BE-\u17C5]
	| '\u17C6'
	| [\u17C7-\u17C8]
	| [\u17C9-\u17D3]
	| '\u17DD'
	| [\u17E0-\u17E9]
	| [\u180B-\u180D]
	| [\u1810-\u1819]
	| '\u18A9'
	| [\u1920-\u1922]
	| [\u1923-\u1926]
	| [\u1927-\u1928]
	| [\u1929-\u192B]
	| [\u1930-\u1931]
	| '\u1932'
	| [\u1933-\u1938]
	| [\u1939-\u193B]
	| [\u1946-\u194F]
	| [\u19B0-\u19C0]
	| [\u19C8-\u19C9]
	| [\u19D0-\u19D9]
	| [\u1A17-\u1A18]
	| [\u1A19-\u1A1B]
	| [\u1DC0-\u1DC3]
	| [\u203F-\u2040]
	| '\u2054'
	| [\u20D0-\u20DC]
	| '\u20E1'
	| [\u20E5-\u20EB]
	| [\u302A-\u302F]
	| [\u3099-\u309A]
	| '\uA802'
	| '\uA806'
	| '\uA80B'
	| [\uA823-\uA824]
	| [\uA825-\uA826]
	| '\uA827'
	| '\uFB1E'
	| [\uFE00-\uFE0F]
	| [\uFE20-\uFE23]
	| [\uFE33-\uFE34]
	| [\uFE4D-\uFE4F]
	| [\uFF10-\uFF19]
	| '\uFF3F';
