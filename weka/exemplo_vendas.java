public class TesteWeka
{
	public static void main(String[] args) throws Exceptions
	{
		DataSource da = new DataSource("vendas.arff");
		Instances ins = ds.getDataSet();
		
		ins.setClassIndex(3); //qual dos atributos é a classe.
		
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(ins);
		
		Instance novo = new DenseInstance(4);
		novo.setDataset(ins);
		novo.setValue(0,"M"); //sexo
		novo.setValue(1,"20-30"); //idade
		novo.setValue(2,"Sim"); //filhos
		
		double probabilidade[] = nb.distributionForInstance(novo);
		System.out.println("Sim: "+probabilidade[1]);
		System.out.println("Não: "+probabilidade[0]);
	}
}