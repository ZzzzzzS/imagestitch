#include "QtGuiApplication1.h"
#include <iostream>
#include <qdebug.h>
#include <qfile.h>
#include <qfileinfo.h>
#include <qfiledialog.h>
#include <qmessagebox.h>
#include <opencv2/opencv.hpp>
#include "ImageStitching.h"

using namespace cv;
using namespace std;


QtGuiApplication1::QtGuiApplication1(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	QObject::connect(ui.filebutton, SIGNAL(clicked()), this, SLOT(AddFileSlot()));
	QObject::connect(ui.pushButton, SIGNAL(clicked()), this, SLOT(ComputeSlot()));
}


void QtGuiApplication1::ComputeSlot()
{
	if (ui.lineEdit->text() == "0")
		QMessageBox::information(this, "error", "add image", QMessageBox::Close);

	Mat temp,final;
	final = imread(filename.at(0).toStdString());
	for (int i = 1; i < filename.size(); i++)
	{
		temp = imread(filename.at(i).toStdString());
		final = ImageStitching(final, temp,false);
	}
	namedWindow("OptimizeImage", 1);
	imshow("OptimizeImage", final);
	waitKey();
}

void QtGuiApplication1::AddFileSlot()
{
	filename.push_back( QFileDialog::getOpenFileName(this, tr("Open File"), QDir::currentPath(), tr("*.jpg *.png")));
	ui.lineEdit->setText(QString::number(filename.size()));
}