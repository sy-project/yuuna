#pragma once

class Transform
{
public:
	static bool isAxisDraw;

	bool isActive;

	string tag;

	cVector3 position;
	cVector3 rotation;
	cVector3 scale;	
protected:
	cVector3 globalPosition;
	cVector3 globalRotation;
	cVector3 globalScale;

	cVector3 pivot;

	Matrix world;
	Matrix* parent;

	MatrixBuffer* worldBuffer;

private:
	Material* material;
	Mesh* mesh;

	MatrixBuffer* transformBuffer;

	vector<VertexColor> vertices;
	vector<UINT> indices;
public:
	Transform(string tag = "Untagged");
	virtual ~Transform();

	void UpdateWorld();
	void RenderAxis();

	void SetWorldBuffer(UINT slot = 0);

	Matrix* GetWorld() { return &world; }
	void SetParent(Matrix* value) { parent = value; }

	cVector3 Forward();
	cVector3 Up();
	cVector3 Right();

	cVector3 GlobalPos() { return globalPosition; }
	cVector3 GlobalRot() { return globalRotation; }
	cVector3 GlobalScale() { return globalScale; }

private:
	void CreateAxis();
};