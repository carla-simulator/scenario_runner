# 模块介绍

## 一、Antlr4简介
Antlr4是一款强大的语法分析器生成工具。给出一门预言的语法，Antlr4能够为该语法生成语法分析器，并自动建立语法分析树。通过遍历此树，并在所需的结构处触发回调函数来实现更多功能。为此，Antlr4提供了自动生成语法分析树遍历器：监听器(listener)或者访问器(visitor)。此外，Antlr4还提供了错误报告和错误功能，并且可以通过实现Antlr4提供的接口来定制错误消息。更多介绍请参考[Antlr官网](https://www.antlr.org/)

### 构建语法分析树

```
antlr4 -Dlanguage=Python3 Example.g4
```
执行此命令后，会生成多个文件，需要了解以下几个文件

- ExampleParser.py：该文件包含一个语法分析器，用于识别‘Example语法’。
- ExampleListener.py：该文件包含了一个以监听器方式遍历语法分析树的遍历器——ExampleListener类。使用监听器遍历语法分析树时，会触发该监听器的回调方法。该类定义了一系列的回调方法，并为这些方法提供默认的空实现，使用时用户只需对需要的方法进行覆盖。
- Example.tokens：Antlr会为语法文件中定义的每个词法符号定义一个对应的数字，该文件用于存储它们之间的对应关系。

### 遍历语法分析树
Antlr提供了两种机制来访问语法解析树，分别是监听器(listener)或者访问器(visitor)。两者最大的区别在于，监听器方法会在访问语法解析树子结点的过程中被自动调用，而访问器方法必须显示调用visit方法来访问子结点，如果没有调用那么子树将不会被访问。

#### listener

ExampleListener类对应Antlr4生成的listener，用户构造自己的访问器时需要继承该类，实现所需的方法。listener会为每个子结点生成一个enterxxx()方法和exitxxx()方法，进入一个结点时会调用enterxxx()方法，当遍历完该结点的所有子节点，从结点退出时，会调用该结点对应的exitxxx()方法。


#### visitor
Antlr4默认生成listener，通过添加-visitor参数指定生成visitor。

```
antlr4 -Dlanguage=Python3 -visitor Example.g4
```


## 二、error_manager模块
Antlr4能够检查并报告输入文件中的词法和语法错误，默认情况下将错误信息送至standard error。为了方便定位和解决错误，项目对错误消息进行定制。
- 监听错误：可以通过实现ErrorListener接口来改变消息的内容和输出目标。项目实现了ErrorListener接口，构建自己的错误监听器，从而将错误信息标准化。下图为项目对ANTLR4错误监听器的扩展
![输入图片说明](developer_documentationimage.png)
- 输出错误信息：使用python的logging模块输出错误信息。


为使自定义的错误管理模块生效，需要在对输入文件进行解析之前，将默认的错误监听器移除，增加自定义的错误监听器。

```
lexer/parser.removeErrorListener()
lexer/parser.addErrorListener(OscErrorListener)
```

## 三、ast_manager模块
ast_manager模块用于创建抽象语法树。包括ast结点定义和ast构建。
- ast结点定义：Node类是所有ast结点的基类，该类定义了1）结点对应的源码的位置信息；2）结点所在的作用域；3）子结点。其它结点均直接或间接继承该结点。
- ast构建：通过遍历Antlr4生成的语法解析树来构建ast。使用listener遍历语法解析树，在回调函数中创建ast结点，并将结点添加到抽象语法树上。

## 四、symbol_manager模块
符号管理模块来记录符号的定义和追踪符号的使用情况，确保符号在使用之前被定义，并且在符号的作用域之内被使用。
- 符号类定义：Symbol类是所有符号的基类，该类定义了符号的基本信息，例如名称和类别，其它符号类在Symbol类的基础上进行扩展。
- 作用域：通过遍历语法解析树，将符号作用域创建为树的结构，在此基础上实现OpenScenario2.0的继承和扩展特性。


