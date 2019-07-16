#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_QtGuiApplication1.h"
#include <iostream>
#include <qstring.h>
using namespace std;

class QtGuiApplication1 : public QMainWindow
{
	Q_OBJECT

private slots:
	void AddFileSlot();
	void ComputeSlot();

public:
	QtGuiApplication1(QWidget *parent = Q_NULLPTR);

private:
	Ui::QtGuiApplication1Class ui;

	vector<QString> filename;
};


#if _MSC_VER >= 1600
#pragma execution_character_set("utf-8")
#endif